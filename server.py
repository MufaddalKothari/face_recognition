import os
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from engine import FaceSearchEngine

app = Flask(__name__)
CORS(app)  // Allows Electron to talk to Flask without security blocks

engine = FaceSearchEngine()

# Global state to track building progress
status = {
    "is_building": False,
    "done": 0,
    "total": 0,
    "last_error": None
}

def progress_updater(done, total):
    status["done"] = done
    status["total"] = total

@app.route('/build', methods=['POST'])
def build_endpoint():
    """Starts the indexing process in a background thread."""
    data = request.json
    folder_path = data.get('folder')

    if not folder_path or not os.path.exists(folder_path):
        return jsonify({"error": "Invalid folder path"}), 400

    if status["is_building"]:
        return jsonify({"error": "Build already in progress"}), 409

    def run_build():
        status["is_building"] = True
        try:
            engine.build_index(folder_path, progress_callback=progress_updater)
        except Exception as e:
            status["last_error"] = str(e)
        finally:
            status["is_building"] = False

    threading.Thread(target=run_build, daemon=True).start()
    return jsonify({"message": "Build started"})

@app.route('/build_status', methods=['GET'])
def get_status():
    """Endpoint for Electron to poll for progress updates."""
    return jsonify(status)

@app.route('/search', methods=['POST'])
def search_endpoint():
    """Returns matching image paths based on query and threshold."""
    data = request.json
    query_path = data.get('query_path')
    threshold = float(data.get('threshold', 0.5))

    if not query_path or not os.path.exists(query_path):
        return jsonify({"error": "Query image not found"}), 400

    try:
        results = engine.search(query_path, threshold=threshold)
        # Convert numpy bboxes to lists so they are JSON serializable
        serialized_results = []
        for r in results:
            serialized_results.append({
                "path": r["path"],
                "score": r["score"],
                "bbox": r["bbox"].tolist() 
            })
        return jsonify(serialized_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Preload the model once when server starts
    engine.preload_model()
    app.run(port=5000, debug=False)
