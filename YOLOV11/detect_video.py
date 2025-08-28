from ultralytics import YOLO
import torch

MODEL_PATH = "runs/detect2/train/weights/best.pt"
VIDEO_PATH = "videos/0f8d2d83-849f-4bf0-be97-2b7231107441.mp4"
# เลือกอุปกรณ์อัตโนมัติ (ถ้ามี CUDA ก็ใช้ GPU)
device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO(MODEL_PATH)

# แสดงผลแบบสดบนจอ (ไม่บันทึกไฟล์) — ไม่มีการแตะต้อง rotation ใดๆ
model.predict(
    source=VIDEO_PATH,
    conf=0.7,
    device=device,
    show=True,        # เปิดหน้าต่าง imshow
    verbose=True
)
