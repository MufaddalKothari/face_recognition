import customtkinter as ctk
from tkinter import filedialog
import threading
import os
import shutil
import platform
import subprocess
import cv2
from PIL import Image
from engine import FaceSearchEngine
import multiprocessing

os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser('~'), '.matplotlib_facesearch')

# -------------------------
# CONFIG
# -------------------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

engine = FaceSearchEngine()

build_running = False
cancel_flag = threading.Event()
results_data = []


# -------------------------
# UTIL
# -------------------------
def open_image(path):
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["open", path])
        elif system == "Windows":
            os.startfile(path)
        else:
            subprocess.run(["xdg-open", path])
    except:
        pass


# -------------------------
# APP
# -------------------------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Face Search")
        self.geometry("1150x780")

        self.logs = []

        self.build_ui()

        # preload model
        threading.Thread(target=engine.preload_model, daemon=True).start()
        self.log("Preloading model in background...")

    # -------------------------
    # UI
    # -------------------------
    def build_ui(self):
        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # -------------------------
        # DATASET SECTION
        # -------------------------
        dataset_frame = ctk.CTkFrame(container)
        dataset_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(dataset_frame, text="Dataset Folder").pack(anchor="w", padx=10, pady=(8, 0))

        self.dataset_entry = ctk.CTkEntry(dataset_frame, placeholder_text="Select dataset folder...")
        self.dataset_entry.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(
            dataset_frame,
            text="Browse Folder",
            command=self.select_dataset,
            corner_radius=10
        ).pack(pady=5)

        self.build_btn = ctk.CTkButton(
            dataset_frame,
            text="Build Dataset",
            command=self.start_build,
            corner_radius=12
        )
        self.build_btn.pack(pady=8)

        self.cancel_btn = ctk.CTkButton(
            dataset_frame,
            text="Cancel Build",
            fg_color="#d9534f",
            hover_color="#c9302c",
            command=self.cancel_build,
            corner_radius=12
        )

        self.progress = ctk.CTkProgressBar(dataset_frame)
        self.progress.pack(fill="x", padx=10, pady=10)
        self.progress.set(0)

        # -------------------------
        # QUERY SECTION
        # -------------------------
        query_frame = ctk.CTkFrame(container)
        query_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(query_frame, text="Query Image").pack(anchor="w", padx=10, pady=(8, 0))

        self.query_entry = ctk.CTkEntry(query_frame, placeholder_text="Select query image...")
        self.query_entry.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(
            query_frame,
            text="Browse Image",
            command=self.select_query,
            corner_radius=10
        ).pack(pady=5)

        ctk.CTkButton(
            query_frame,
            text="Search",
            command=self.search,
            corner_radius=12
        ).pack(pady=8)

        # -------------------------
        # ✅ THRESHOLD SLIDER (NEW)
        # -------------------------
        self.threshold_label = ctk.CTkLabel(
            query_frame,
            text="Match Threshold: 0.50"
        )
        self.threshold_label.pack(pady=(5, 0))

        self.threshold_slider = ctk.CTkSlider(
            query_frame,
            from_=0.2,
            to=0.9,
            number_of_steps=70,
            command=self.update_threshold_label
        )
        self.threshold_slider.set(0.5)
        self.threshold_slider.pack(fill="x", padx=20, pady=(0, 10))

        # -------------------------
        # ACTIONS
        # -------------------------
        actions = ctk.CTkFrame(container)
        actions.pack(fill="x", pady=5)

        ctk.CTkButton(actions, text="Select All", command=self.select_all, corner_radius=10)\
            .pack(side="left", padx=10)

        ctk.CTkButton(actions, text="Save Selected", command=self.save_selected, corner_radius=10)\
            .pack(side="left", padx=10)

        ctk.CTkButton(actions, text="Show Logs", command=self.show_logs, corner_radius=10)\
            .pack(side="right", padx=10)

        # -------------------------
        # RESULTS
        # -------------------------
        self.results_frame = ctk.CTkScrollableFrame(container)
        self.results_frame.pack(fill="both", expand=True, pady=10)

    # -------------------------
    # FILE PICKERS
    # -------------------------
    def select_dataset(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_entry.delete(0, "end")
            self.dataset_entry.insert(0, path)

    def select_query(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")]
        )
        if path:
            self.query_entry.delete(0, "end")
            self.query_entry.insert(0, path)

    # -------------------------
    # BUILD
    # -------------------------
    def start_build(self):
        global build_running

        if build_running:
            return

        build_running = True
        cancel_flag.clear()

        self.progress.set(0)
        self.build_btn.pack_forget()
        self.cancel_btn.pack(pady=8)

        threading.Thread(target=self.run_build, daemon=True).start()

    def run_build(self):
        path = self.dataset_entry.get()

        self.log("Starting build...")

        engine.build_index(
            path,
            cancel_flag=cancel_flag,
            progress_callback=self.update_progress
        )

        self.after(0, self.restore_ui)

    def cancel_build(self):
        global build_running

        cancel_flag.set()
        engine.reset_index()

        self.log("Build cancelled")

        build_running = False
        self.restore_ui()

    def restore_ui(self):
        global build_running

        build_running = False
        self.cancel_btn.pack_forget()
        self.build_btn.pack(pady=8)

    def update_progress(self, done, total):
        if total == 0:
            return

        self.progress.set(min(1.0, done / total))
        self.log(f"{done}/{total}")

    # -------------------------
    # SEARCH
    # -------------------------
    def search(self):
        path = self.query_entry.get()
        if not path:
            return

        threading.Thread(target=self.run_search, args=(path,), daemon=True).start()

    def run_search(self, path):
        global results_data

        self.log("Searching...")

        threshold = self.threshold_slider.get()  # ✅ NEW

        results = engine.search(
            path,
            threshold=threshold   # ✅ PASSED
        )

        results_data = [
            {**r, "selected": ctk.BooleanVar(value=False)}
            for r in results
        ]

        self.after(0, self.display_results)

    # -------------------------
    # RESULTS
    # -------------------------
    def display_results(self):
        for w in self.results_frame.winfo_children():
            w.destroy()

        cols = 5

        for i, r in enumerate(results_data):
            card = ctk.CTkFrame(self.results_frame, corner_radius=12)
            card.grid(row=i // cols, column=i % cols, padx=10, pady=10)

            img = cv2.imread(r["path"])
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            img = cv2.resize(img, (140, 140))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            im = ctk.CTkImage(light_image=Image.fromarray(img), size=(140, 140))

            lbl = ctk.CTkLabel(card, image=im, text="")
            lbl.image = im
            lbl.pack(pady=5)

            ctk.CTkCheckBox(card, variable=r["selected"], text="Select").pack()

            ctk.CTkButton(
                card,
                text="Preview",
                command=lambda p=r["path"]: open_image(p),
                corner_radius=10
            ).pack(pady=5)

    # -------------------------
    # ACTIONS
    # -------------------------
    def select_all(self):
        for r in results_data:
            r["selected"].set(True)

    def save_selected(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        for r in results_data:
            if r["selected"].get():
                shutil.copy(r["path"], folder)

        self.log("Saved selected images")

    # -------------------------
    # LOGS
    # -------------------------
    def log(self, msg):
        self.logs.append(msg)

    def show_logs(self):
        win = ctk.CTkToplevel(self)
        win.title("Logs")
        win.geometry("600x400")

        box = ctk.CTkTextbox(win)
        box.pack(fill="both", expand=True)

        box.insert("end", "\n".join(self.logs))

    # -------------------------
    # SLIDER
    # -------------------------
    def update_threshold_label(self, value):
        self.threshold_label.configure(
            text=f"Match Threshold: {value:.2f}"
        )


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = App()
    app.mainloop()
