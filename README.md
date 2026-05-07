# Morphology Demo (Erosion / Dilation / Opening / Closing)

Demo UI xử lý ảnh hình thái học theo tài liệu:

- Erosion
- Dilation
- Opening = Erosion + Dilation
- Closing = Dilation + Erosion

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy chương trình

### UI OpenCV (nhanh, chạy trực tiếp Python)

- Camera realtime:

```bash
python opencv_ui.py
```

- Ảnh (upload bằng đường dẫn):

```bash
python opencv_ui.py --image "path\to\image.jpg"
```

Phím tắt:

- `a`: xem mosaic nhiều khung
- `1/2/3/4`: focus Erosion/Dilation/Opening/Closing
- `i`: invert threshold
- `r/e/c`: kernel shape Rect/Ellipse/Cross
- `s`: lưu ảnh vào thư mục `outputs/`
- `q` hoặc `Esc`: thoát

### Cách nhanh (khuyên dùng)

- Windows PowerShell:

```bash
.\run.ps1
```

- Windows CMD:

```bash
run.cmd
```

- Linux/macOS:

```bash
chmod +x run.sh
./run.sh
```

### Cách thủ công

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tính năng

- **Upload ảnh**: hiển thị Original/Gray/Binary và các kết quả morphology.
- **Camera realtime**: xử lý webcam theo thời gian thực, chọn view (Binary/Erosion/Closing...).

