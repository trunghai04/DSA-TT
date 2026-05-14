import base64
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np


def _ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _white_ratio(binary: np.ndarray) -> float:
    return float(np.count_nonzero(binary)) / float(binary.size)


def to_binary(gray: np.ndarray, thresh: int, invert: bool, method: str, auto_polarity: bool = True) -> np.ndarray:
    if method == "Adaptive Gaussian":
        normal = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        inverted = cv2.bitwise_not(normal)
        if auto_polarity:
            return inverted if _white_ratio(inverted) < _white_ratio(normal) else normal
        return inverted if invert else normal
    if method == "Otsu":
        _, normal = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = cv2.bitwise_not(normal)
        if auto_polarity:
            return inverted if _white_ratio(inverted) < _white_ratio(normal) else normal
        return inverted if invert else normal
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, thresh, 255, mode)
    return binary


def make_kernel(size: int, shape_name: str) -> np.ndarray:
    k = _ensure_odd(int(size))
    shape_map = {
        "Rect": cv2.MORPH_RECT,
        "Ellipse": cv2.MORPH_ELLIPSE,
        "Cross": cv2.MORPH_CROSS,
    }
    shape = shape_map.get(shape_name, cv2.MORPH_RECT)
    return cv2.getStructuringElement(shape, (k, k))


def apply_morphology(binary: np.ndarray, kernel: np.ndarray, iterations: int):
    it = max(1, int(iterations))
    erosion = cv2.erode(binary, kernel, iterations=it)
    dilation = cv2.dilate(binary, kernel, iterations=it)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=it)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=it)
    return erosion, dilation, opening, closing


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def binary_to_rgb(binary: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def add_salt_pepper(binary: np.ndarray, amount_pct: int, salt_vs_pepper: int) -> np.ndarray:
    amt = max(0, min(30, int(amount_pct)))
    if amt == 0:
        return binary
    p = amt / 100.0
    rnd = np.random.random(binary.shape)
    salt_p = p * (max(0, min(100, int(salt_vs_pepper))) / 100.0)
    pepper_p = p - salt_p
    out = binary.copy()
    out[rnd < pepper_p] = 0
    out[(rnd >= pepper_p) & (rnd < pepper_p + salt_p)] = 255
    return out


def preprocess_gray(gray: np.ndarray, blur_method: str, blur_size: int) -> np.ndarray:
    k = _ensure_odd(max(1, int(blur_size)))
    if blur_method == "Median Blur":
        return cv2.medianBlur(gray, k)
    if blur_method == "Gaussian Blur":
        return cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def _label_bgr(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _grid_2x3(frames_bgr: list[np.ndarray]) -> np.ndarray:
    h0, w0 = frames_bgr[0].shape[:2]
    cells = [cv2.resize(f, (w0, h0), interpolation=cv2.INTER_AREA) for f in frames_bgr]
    row1 = np.hstack(cells[0:3])
    row2 = np.hstack(cells[3:6])
    return np.vstack([row1, row2])


def resize_keep_aspect(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    new_h = int(h * (max_w / w))
    return cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_AREA)


def fit_thumbnail(img: np.ndarray, max_w: int = 320, max_h: int = 220) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def process_frame_bgr(
    frame_bgr: np.ndarray,
    blur_method: str,
    blur_size: int,
    thresh: int,
    thresh_method: str,
    invert: bool,
    kernel_size: int,
    kernel_shape: str,
    iterations: int,
    show: str,
    noise_amount_pct: int,
    salt_vs_pepper: int,
) -> np.ndarray:
    gray = to_gray(frame_bgr)
    gray = preprocess_gray(gray, blur_method=blur_method, blur_size=blur_size)
    binary = to_binary(gray, thresh=thresh, invert=invert, method=thresh_method, auto_polarity=True)
    binary = add_salt_pepper(binary, amount_pct=noise_amount_pct, salt_vs_pepper=salt_vs_pepper)
    kernel = make_kernel(kernel_size, kernel_shape)
    erosion, dilation, opening, closing = apply_morphology(binary, kernel, iterations)
    closing_then_opening = apply_morphology(closing, kernel, iterations)[2]

    view_map = {
        "Original": bgr_to_rgb(frame_bgr),
        "Gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        "Binary": binary_to_rgb(binary),
        "Erosion": binary_to_rgb(erosion),
        "Dilation": binary_to_rgb(dilation),
        "Opening": binary_to_rgb(opening),
        "Closing": binary_to_rgb(closing),
        "Result": binary_to_rgb(closing_then_opening),
    }
    if show == "Compare (2x3)":
        bgrs = [
            _label_bgr(cv2.cvtColor(view_map["Binary"], cv2.COLOR_RGB2BGR), "Binary"),
            _label_bgr(cv2.cvtColor(view_map["Opening"], cv2.COLOR_RGB2BGR), "Opening"),
            _label_bgr(cv2.cvtColor(view_map["Closing"], cv2.COLOR_RGB2BGR), "Closing"),
            _label_bgr(cv2.cvtColor(view_map["Result"], cv2.COLOR_RGB2BGR), "Final Result"),
            _label_bgr(cv2.cvtColor(view_map["Gray"], cv2.COLOR_RGB2BGR), "Gray"),
            _label_bgr(cv2.cvtColor(view_map["Original"], cv2.COLOR_RGB2BGR), "Original"),
        ]
        grid_bgr = _grid_2x3(bgrs)
        return cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)
    return view_map.get(show, bgr_to_rgb(frame_bgr))


def image_to_tk(img: np.ndarray) -> tk.PhotoImage:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Không thể chuyển ảnh sang PNG")
    data = base64.b64encode(buf.tobytes())
    return tk.PhotoImage(data=data)


class MorphologyApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Morphology Demo - Desktop UI")
        self.root.geometry("1400x900")

        self.cap = None
        self.camera_running = False
        self.current_bgr = None
        self.result_refs = []
        self.result_images = []
        self.camera_ref = None
        self.fullscreen_window = None
        self.fullscreen_image_ref = None
        self.status_ref = None

        self.blur_method = tk.StringVar(value="Median Blur")
        self.blur_size = tk.IntVar(value=5)
        self.thresh_method = tk.StringVar(value="Otsu")
        self.thresh = tk.IntVar(value=127)
        self.invert = tk.BooleanVar(value=False)
        self.kernel_shape = tk.StringVar(value="Ellipse")
        self.kernel_size = tk.IntVar(value=3)
        self.iterations = tk.IntVar(value=1)
        self.show = tk.StringVar(value="Compare (2x3)")
        self.overlay = tk.BooleanVar(value=True)
        self.noise_amount_pct = tk.IntVar(value=0)
        self.salt_vs_pepper = tk.IntVar(value=50)

        self._build_ui()
    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, padding=12)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="Morphology controls", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        self._add_combo(sidebar, "Blur method", self.blur_method, ["None", "Gaussian Blur", "Median Blur"], 1)
        self._add_slider(sidebar, "Blur size", self.blur_size, 1, 31, 2, 2)
        self._add_combo(sidebar, "Threshold method", self.thresh_method, ["Fixed", "Adaptive Gaussian", "Otsu"], 3)
        self._add_slider(sidebar, "Threshold", self.thresh, 0, 255, 1, 4)
        self._add_check(sidebar, "Invert (đổi trắng/đen)", self.invert, 5)
        self._add_combo(sidebar, "Kernel shape", self.kernel_shape, ["Rect", "Ellipse", "Cross"], 6)
        self._add_slider(sidebar, "Kernel size (odd)", self.kernel_size, 1, 31, 2, 7)
        self._add_slider(sidebar, "Iterations", self.iterations, 1, 5, 1, 8)

        ttk.Separator(sidebar).grid(row=9, column=0, sticky="ew", pady=12)
        self._add_combo(
            sidebar,
            "View",
            self.show,
            ["Compare (2x3)", "Original", "Gray", "Binary", "Erosion", "Dilation", "Opening", "Closing", "Result"],
            10,
        )
        self._add_check(sidebar, "Overlay thông số", self.overlay, 11)
        self._add_slider(sidebar, "Noise (%)", self.noise_amount_pct, 0, 20, 1, 12)
        self._add_slider(sidebar, "Salt vs Pepper", self.salt_vs_pepper, 0, 100, 5, 13)
        ttk.Button(sidebar, text="Cập nhật ảnh đang mở", command=self.refresh_views).grid(row=14, column=0, sticky="ew", pady=(10, 4))

        ttk.Separator(sidebar).grid(row=15, column=0, sticky="ew", pady=12)

        ttk.Button(sidebar, text="Upload ảnh", command=self.load_image).grid(row=16, column=0, sticky="ew", pady=(12, 4))
        ttk.Button(sidebar, text="Bắt đầu camera", command=self.start_camera).grid(row=17, column=0, sticky="ew", pady=4)
        ttk.Button(sidebar, text="Dừng camera", command=self.stop_camera).grid(row=18, column=0, sticky="ew", pady=4)
        ttk.Button(sidebar, text="Thoát", command=self.on_close).grid(row=19, column=0, sticky="ew", pady=(20, 4))

        self.tabs = ttk.Notebook(self.root)
        self.tabs.grid(row=0, column=1, sticky="nsew")

        self.tab_upload = ttk.Frame(self.tabs, padding=12)
        self.tab_camera = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.tab_upload, text="Upload ảnh")
        self.tabs.add(self.tab_camera, text="Camera realtime")

        self._build_upload_tab()
        self._build_camera_tab()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _add_slider(self, parent, label, var, start, end, step, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=4)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        ttk.Scale(
            frame,
            from_=start,
            to=end,
            orient="horizontal",
            command=lambda v, x=var, s=step: x.set(int(round(float(v) / s) * s)),
        ).grid(row=1, column=0, sticky="ew")
        ttk.Label(frame, textvariable=var).grid(row=1, column=1, padx=(8, 0))

    def _add_check(self, parent, label, var, row):
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, sticky="w", pady=4)

    def _add_combo(self, parent, label, var, values, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=4)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(frame, textvariable=var, values=values, state="readonly")
        combo.grid(row=1, column=0, sticky="ew")
        return combo

    def _build_upload_tab(self) -> None:
        self.upload_info = ttk.Label(self.tab_upload, text="Chọn ảnh để xem từng kết quả riêng trong 6 ô bên dưới.")
        self.upload_info.pack(anchor="w", pady=(0, 10))

        self.status_ref = ttk.Label(self.tab_upload, text="Mẹo: click vào ảnh để phóng lớn fullscreen.")
        self.status_ref.pack(anchor="w", pady=(0, 10))

        self.upload_grid = ttk.LabelFrame(self.tab_upload, text="Kết quả riêng lẻ", padding=10)
        self.upload_grid.pack(fill="both", expand=True, pady=10)
        self._create_result_labels(self.upload_grid)

    def _build_camera_tab(self) -> None:
        self.camera_status = ttk.Label(self.tab_camera, text="Camera chưa chạy.")
        self.camera_status.pack(anchor="w", pady=(0, 10))
        self.camera_frame = ttk.LabelFrame(self.tab_camera, text="Camera output", padding=10)
        self.camera_frame.pack(fill="both", expand=True)
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack()

    def _create_result_labels(self, parent) -> None:
        for child in parent.winfo_children():
            child.destroy()
        self.result_refs = []
        self.result_images = []
        for r in range(2):
            parent.rowconfigure(r, weight=1)
            for c in range(3):
                parent.columnconfigure(c, weight=1)
                cell = ttk.Frame(parent, padding=4, relief="ridge")
                cell.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
                cell.columnconfigure(0, weight=1)
                cell.rowconfigure(0, weight=1)
                lbl = ttk.Label(cell, anchor="center")
                lbl.grid(row=0, column=0, sticky="nsew")
                lbl.bind("<Enter>", lambda e, w=cell: w.configure(relief="solid"))
                lbl.bind("<Leave>", lambda e, w=cell: w.configure(relief="ridge"))
                self.result_refs.append(lbl)
                self.result_images.append(None)

    def _update_upload_results(self, img_bgr: np.ndarray) -> None:
        gray = to_gray(img_bgr)
        gray = preprocess_gray(gray, blur_method=self.blur_method.get(), blur_size=self.blur_size.get())
        binary = to_binary(gray, thresh=self.thresh.get(), invert=self.invert.get(), method=self.thresh_method.get(), auto_polarity=True)
        kernel = make_kernel(self.kernel_size.get(), self.kernel_shape.get())
        erosion, dilation, opening, closing = apply_morphology(binary, kernel, self.iterations.get())
        result = apply_morphology(closing, kernel, self.iterations.get())[2]

        images = [
            (bgr_to_rgb(img_bgr), "Original"),
            (cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "Gray"),
            (binary, "Binary"),
            (erosion, "Erosion"),
            (dilation, "Dilation"),
            (opening, "Opening"),
            (closing, "Closing"),
            (result, "Result"),
        ]

        grid_imgs = [
            images[2][0],
            images[5][0],
            images[6][0],
            images[3][0],
            images[4][0],
            images[0][0],
        ]
        grid_names = ["Binary", "Opening", "Closing", "Erosion", "Dilation", "Original"]
        self.result_images = grid_imgs
        for idx, (lbl, img, name) in enumerate(zip(self.result_refs, grid_imgs, grid_names)):
            lbl.configure(text=name, compound="top")
            tk_img = image_to_tk(fit_thumbnail(img, 360, 220))
            lbl.image = tk_img
            lbl.configure(image=tk_img)
            lbl.bind("<Button-1>", lambda e, i=idx, n=name: self.open_fullscreen(i, n))
            lbl.configure(cursor="hand2")

    def load_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh đã chọn.")
            return
        self.current_bgr = resize_keep_aspect(img_bgr, max_w=1200)
        self._update_upload_results(self.current_bgr)
        self.tabs.select(self.tab_upload)

    def start_camera(self) -> None:
        if self.camera_running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được webcam.")
            self.cap = None
            return
        self.camera_running = True
        self.camera_status.configure(text="Camera đang chạy...")
        self._camera_loop()

    def stop_camera(self) -> None:
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_status.configure(text="Camera đã dừng.")

    def _camera_loop(self) -> None:
        if not self.camera_running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            frame = resize_keep_aspect(frame, max_w=1100)
            out_rgb = process_frame_bgr(
                frame,
                self.blur_method.get(),
                self.blur_size.get(),
                self.thresh.get(),
                self.thresh_method.get(),
                self.invert.get(),
                self.kernel_size.get(),
                self.kernel_shape.get(),
                self.iterations.get(),
                self.show.get(),
                self.noise_amount_pct.get(),
                self.salt_vs_pepper.get(),
            )
            if self.overlay.get():
                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                text = (
                    f"{self.show.get()} | blur={self.blur_method.get()}:{_ensure_odd(self.blur_size.get())} "
                    f"thr={self.thresh_method.get()} inv={int(self.invert.get())} "
                    f"k={self.kernel_shape.get()}:{_ensure_odd(self.kernel_size.get())} it={self.iterations.get()}"
                )
                cv2.rectangle(out_bgr, (0, 0), (out_bgr.shape[1], 32), (0, 0, 0), -1)
                cv2.putText(out_bgr, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            out_rgb = fit_thumbnail(out_rgb, 1100, 700)
            self.camera_ref = image_to_tk(out_rgb)
            self.camera_label.configure(image=self.camera_ref)
        self.root.after(30, self._camera_loop)

    def refresh_views(self) -> None:
        if self.current_bgr is not None:
            self._update_upload_results(self.current_bgr)

    def open_fullscreen(self, index: int, title: str) -> None:
        if index >= len(self.result_images):
            return
        img = self.result_images[index]
        if img is None:
            return

        if self.fullscreen_window is not None and self.fullscreen_window.winfo_exists():
            self.fullscreen_window.destroy()

        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg="black")
        win.attributes("-fullscreen", True)
        win.bind("<Escape>", lambda e: win.destroy())
        win.bind("<Button-1>", lambda e: win.destroy())

        container = tk.Frame(win, bg="black")
        container.pack(fill="both", expand=True)
        container.pack_propagate(False)

        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
        display = fit_thumbnail(img, max_w=screen_w - 40, max_h=screen_h - 80)
        self.fullscreen_image_ref = image_to_tk(display)

        lbl = tk.Label(container, image=self.fullscreen_image_ref, bg="black")
        lbl.pack(expand=True)

        footer = tk.Label(
            win,
            text=f"{title}  |  Nhấn Esc hoặc click để đóng",
            fg="white",
            bg="black",
            font=("Segoe UI", 12, "bold"),
        )
        footer.pack(side="bottom", pady=10)
        self.fullscreen_window = win

    def on_close(self) -> None:
        self.stop_camera()
        if self.fullscreen_window is not None and self.fullscreen_window.winfo_exists():
            self.fullscreen_window.destroy()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    MorphologyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

