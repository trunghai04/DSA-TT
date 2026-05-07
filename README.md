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

