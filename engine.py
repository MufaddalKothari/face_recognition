import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

import cv2
import numpy as np
import faiss
import insightface
from multiprocessing import Pool, cpu_count
import sys
import os

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# -------------------------
# GLOBAL (per worker)
# -------------------------
face_app = None

# -------------------------
# WORKER INIT
# -------------------------
def init_worker():
    global face_app
    model_path = get_resource_path('.')
    face_app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'], root=model_path, name="buffalo_l")
    face_app.prepare(ctx_id=0)


# -------------------------
# WORKER FUNCTION
# -------------------------
def process_batch(paths):
    global face_app

    embeddings = []
    metadata = []

    for path in paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            max_dim = 1280 # 720p/1080p equivalent is plenty for faces
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            faces = face_app.get(img)

            for face in faces:
                embeddings.append(face.embedding)
                metadata.append({
                    "path": path,
                    "bbox": face.bbox.astype(int)
                })

        except:
            continue

    return embeddings, metadata, len(paths)



# -------------------------
# ENGINE
# -------------------------
class FaceSearchEngine:
    def __init__(self):
        self.index = None
        self.embeddings = []
        self.metadata = []
        self.face_app = None

    def preload_model(self):
        if self.face_app is None:
            print("Preloading model...")
            model_path = get_resource_path('.')
            self.face_app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'], root=model_path, name="buffalo_l")
            self.face_app.prepare(ctx_id=0)
            print("Model ready")

    def get_all_images(self, folder):
        exts = (".jpg", ".jpeg", ".png", ".webp")
        paths = []

        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(exts):
                    paths.append(os.path.join(root, f))

        return paths

    def reset_index(self):
        self.index = None
        self.embeddings = []
        self.metadata = []

    # -------------------------
    # BUILD INDEX
    # -------------------------
    def build_index(self, folder, cancel_flag=None, progress_callback=None):
        self.reset_index()

        paths = self.get_all_images(folder)
        total = len(paths)

        if total == 0:
            return

        num_workers = max(1, cpu_count() - 1)
        chunk_size = max(1, total // num_workers)

        chunks = [paths[i:i + chunk_size] for i in range(0, total, chunk_size)]

        processed_images = 0

        with Pool(num_workers, initializer=init_worker) as pool:
            for embeds, metas, batch_count in pool.imap(process_batch, chunks):

                if cancel_flag and cancel_flag.is_set():
                    pool.terminate()
                    pool.join()
                    return

                self.embeddings.extend(embeds)
                self.metadata.extend(metas)

                processed_images += batch_count

                if progress_callback:
                    progress_callback(processed_images, total)

        if len(self.embeddings) == 0:
            return

        X = np.array(self.embeddings).astype("float32")
        faiss.normalize_L2(X)

        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X)

    # -------------------------
    # SEARCH
    # -------------------------
    def search(self, query_path, top_k=20, threshold=0.5):
        if self.index is None:
            return []

        self.preload_model()

        img = cv2.imread(query_path)
        if img is None:
            return []

        faces = self.face_app.get(img)
        if not faces:
            return []

        emb = faces[0].embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(emb)

        D, I = self.index.search(emb, top_k)

        results = []
        seen = set()

        for score, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue

            if score < threshold:
                continue

            meta = self.metadata[idx]

            if meta["path"] in seen:
                continue
            seen.add(meta["path"])

            results.append({
                "path": meta["path"],
                "bbox": meta["bbox"],
                "score": float(score)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
