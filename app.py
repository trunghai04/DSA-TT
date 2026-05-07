import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


def _ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def to_binary(gray: np.ndarray, thresh: int, invert: bool) -> np.ndarray:
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


def resize_keep_aspect(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    new_h = int(h * (max_w / w))
    return cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_AREA)


def process_frame_bgr(
    frame_bgr: np.ndarray,
    thresh: int,
    invert: bool,
    kernel_size: int,
    kernel_shape: str,
    iterations: int,
    show: str,
) -> np.ndarray:
    gray = to_gray(frame_bgr)
    binary = to_binary(gray, thresh=thresh, invert=invert)
    kernel = make_kernel(kernel_size, kernel_shape)
    erosion, dilation, opening, closing = apply_morphology(binary, kernel, iterations)

    view_map = {
        "Original": bgr_to_rgb(frame_bgr),
        "Gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        "Binary": binary_to_rgb(binary),
        "Erosion": binary_to_rgb(erosion),
        "Dilation": binary_to_rgb(dilation),
        "Opening": binary_to_rgb(opening),
        "Closing": binary_to_rgb(closing),
    }
    return view_map.get(show, bgr_to_rgb(frame_bgr))


def render_static_demo(
    img_bgr: np.ndarray,
    thresh: int,
    invert: bool,
    kernel_size: int,
    kernel_shape: str,
    iterations: int,
):
    gray = to_gray(img_bgr)
    binary = to_binary(gray, thresh=thresh, invert=invert)
    kernel = make_kernel(kernel_size, kernel_shape)
    erosion, dilation, opening, closing = apply_morphology(binary, kernel, iterations)

    cols1 = st.columns(3)
    with cols1[0]:
        st.image(bgr_to_rgb(img_bgr), caption="Original (RGB)", use_container_width=True)
    with cols1[1]:
        st.image(gray, caption="Gray", use_container_width=True, clamp=True)
    with cols1[2]:
        st.image(binary, caption="Binary", use_container_width=True, clamp=True)

    cols2 = st.columns(4)
    with cols2[0]:
        st.image(erosion, caption="Erosion", use_container_width=True, clamp=True)
    with cols2[1]:
        st.image(dilation, caption="Dilation", use_container_width=True, clamp=True)
    with cols2[2]:
        st.image(opening, caption="Opening", use_container_width=True, clamp=True)
    with cols2[3]:
        st.image(closing, caption="Closing", use_container_width=True, clamp=True)


class MorphVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.thresh = 127
        self.invert = False
        self.kernel_size = 5
        self.kernel_shape = "Rect"
        self.iterations = 1
        self.show = "Closing"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        out_rgb = process_frame_bgr(
            img_bgr,
            thresh=self.thresh,
            invert=self.invert,
            kernel_size=self.kernel_size,
            kernel_shape=self.kernel_shape,
            iterations=self.iterations,
            show=self.show,
        )
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")


st.set_page_config(page_title="Morphology Demo", layout="wide")
st.title("Demo Morphology: Erosion • Dilation • Opening • Closing")

with st.sidebar:
    st.subheader("Cấu hình xử lý")
    thresh = st.slider("Threshold", 0, 255, 127, 1)
    invert = st.toggle("Invert (đổi trắng/đen)", value=False)
    kernel_shape = st.selectbox("Kernel shape", ["Rect", "Ellipse", "Cross"], index=0)
    kernel_size = st.slider("Kernel size (odd)", 1, 31, 5, 2)
    iterations = st.slider("Iterations", 1, 5, 1, 1)

tab_upload, tab_camera = st.tabs(["Upload ảnh", "Camera realtime"])

with tab_upload:
    st.write("Tải ảnh lên để xem toàn bộ pipeline: Gray → Threshold → Morphology.")
    up = st.file_uploader("Chọn ảnh (jpg/png)", type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Chưa có ảnh. Hãy upload để xem kết quả.")
    else:
        file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Không đọc được ảnh. Thử ảnh khác giúp mình nhé.")
        else:
            img_bgr = resize_keep_aspect(img_bgr, max_w=1200)
            render_static_demo(
                img_bgr,
                thresh=thresh,
                invert=invert,
                kernel_size=kernel_size,
                kernel_shape=kernel_shape,
                iterations=iterations,
            )

with tab_camera:
    st.write("Bật webcam để xem xử lý theo thời gian thực (realtime).")
    show = st.selectbox(
        "Xem khung nào?",
        ["Original", "Gray", "Binary", "Erosion", "Dilation", "Opening", "Closing"],
        index=6,
    )

    ctx = webrtc_streamer(
        key="morph-demo",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=MorphVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.thresh = thresh
        ctx.video_processor.invert = invert
        ctx.video_processor.kernel_size = kernel_size
        ctx.video_processor.kernel_shape = kernel_shape
        ctx.video_processor.iterations = iterations
        ctx.video_processor.show = show

