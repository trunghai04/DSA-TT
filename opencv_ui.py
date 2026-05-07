import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def make_kernel(size: int, shape: str) -> np.ndarray:
    k = ensure_odd(max(1, int(size)))
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    s = shape_map.get(shape.lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(s, (k, k))


def binarize(gray: np.ndarray, thresh: int, invert: bool) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, int(thresh), 255, mode)
    return binary


def morph_all(binary: np.ndarray, kernel: np.ndarray, iterations: int):
    it = max(1, int(iterations))
    erosion = cv2.erode(binary, kernel, iterations=it)
    dilation = cv2.dilate(binary, kernel, iterations=it)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=it)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=it)
    return erosion, dilation, opening, closing


def put_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def put_footer(img_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    if not lines:
        return img_bgr
    out = img_bgr.copy()
    line_h = 22
    pad = 8
    footer_h = pad * 2 + line_h * len(lines)
    y0 = max(0, out.shape[0] - footer_h)
    cv2.rectangle(out, (0, y0), (out.shape[1], out.shape[0]), (0, 0, 0), thickness=-1)
    y = y0 + pad + 16
    for line in lines:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += line_h
    return out


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def resize_to(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def mosaic_2x3(frames_bgr: list[np.ndarray], cell_w: int, cell_h: int) -> np.ndarray:
    cells = [resize_to(f, cell_w, cell_h) for f in frames_bgr]
    row1 = np.hstack(cells[0:3])
    row2 = np.hstack(cells[3:6])
    return np.vstack([row1, row2])


def build_view(frame_bgr: np.ndarray, thresh: int, invert: bool, ksize: int, kshape: str, iters: int):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray, thresh=thresh, invert=invert)
    kernel = make_kernel(ksize, kshape)
    erosion, dilation, opening, closing = morph_all(binary, kernel, iters)

    views = {
        "Original": frame_bgr,
        "Gray": to_bgr(gray),
        "Binary": to_bgr(binary),
        "Erosion": to_bgr(erosion),
        "Dilation": to_bgr(dilation),
        "Opening": to_bgr(opening),
        "Closing": to_bgr(closing),
    }
    return views


def parse_args():
    p = argparse.ArgumentParser(description="Morphology demo UI with OpenCV.")
    p.add_argument("--image", type=str, default="", help="Path to an image (jpg/png).")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    p.add_argument("--width", type=int, default=1200, help="Max display width for mosaic.")
    p.add_argument("--kernel-shape", type=str, default="rect", choices=["rect", "ellipse", "cross"])
    p.add_argument("--invert", action="store_true", help="Invert threshold (binary inv).")
    return p.parse_args()


def main():
    args = parse_args()

    window = "Morphology Demo (OpenCV)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    state = {
        "thresh": 127,
        "ksize": 5,
        "iters": 1,
        "invert": bool(args.invert),
        "kshape": args.kernel_shape,
        "show_all": True,
        "focus": "Closing",
    }

    def on_change(_=0):
        pass

    cv2.createTrackbar("Threshold", window, state["thresh"], 255, on_change)
    cv2.createTrackbar("Kernel size", window, state["ksize"], 31, on_change)
    cv2.createTrackbar("Iterations", window, state["iters"], 5, on_change)

    def read_trackbar():
        state["thresh"] = cv2.getTrackbarPos("Threshold", window)
        k = cv2.getTrackbarPos("Kernel size", window)
        state["ksize"] = max(1, k)
        state["iters"] = max(1, cv2.getTrackbarPos("Iterations", window))

    img_bgr: np.ndarray | None = None
    cap: cv2.VideoCapture | None = None

    if args.image:
        path = Path(args.image)
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise SystemExit(f"Cannot read image: {path}")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera index {args.camera}")

    last_save_ts = 0.0
    fps_ema: float | None = None
    last_ts = time.time()

    while True:
        now_ts = time.time()
        dt = max(1e-6, now_ts - last_ts)
        last_ts = now_ts
        fps = 1.0 / dt
        fps_ema = fps if fps_ema is None else (0.9 * fps_ema + 0.1 * fps)

        read_trackbar()

        if cap is not None:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                time.sleep(0.01)
                continue
        else:
            frame_bgr = img_bgr.copy()  # type: ignore[union-attr]

        views = build_view(
            frame_bgr,
            thresh=state["thresh"],
            invert=state["invert"],
            ksize=state["ksize"],
            kshape=state["kshape"],
            iters=state["iters"],
        )

        info = (
            f"fps={fps_ema:0.1f}  th={state['thresh']}  inv={int(state['invert'])}  "
            f"kernel={state['kshape']}:{ensure_odd(state['ksize'])}  it={state['iters']}  "
            f"mode={'ALL' if state['show_all'] else state['focus']}"
        )

        if state["show_all"]:
            order = ["Original", "Binary", "Erosion", "Dilation", "Opening", "Closing"]
            frames = [put_label(to_bgr(views[name]), name) for name in order]
            cell_w = max(240, args.width // 3)
            cell_h = int(cell_w * 0.66)
            canvas = mosaic_2x3(frames, cell_w, cell_h)
        else:
            focus = state["focus"]
            canvas = put_label(to_bgr(views[focus]), focus)

        explain = []
        if not state["show_all"] and state["focus"] == "Opening":
            explain = ["Opening = Erosion + Dilation", "Remove small white noise, keep main shape"]
        elif not state["show_all"] and state["focus"] == "Closing":
            explain = ["Closing = Dilation + Erosion", "Fill small holes, connect broken parts"]
        canvas = put_footer(canvas, explain)

        canvas = put_label(canvas, f"{window} | {info}")
        cv2.imshow(window, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("a"):
            state["show_all"] = True
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            state["show_all"] = False
            state["focus"] = {
                ord("1"): "Erosion",
                ord("2"): "Dilation",
                ord("3"): "Opening",
                ord("4"): "Closing",
            }[key]
        if key == ord("i"):
            state["invert"] = not state["invert"]
        if key == ord("r"):
            state["kshape"] = "rect"
        if key == ord("e"):
            state["kshape"] = "ellipse"
        if key == ord("c"):
            state["kshape"] = "cross"
        if key == ord("s"):
            now = time.time()
            if now - last_save_ts > 0.5:
                out = Path("outputs")
                out.mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                name = f"morph_{stamp}_{'all' if state['show_all'] else state['focus'].lower()}.png"
                cv2.imwrite(str(out / name), canvas)
                last_save_ts = now

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

