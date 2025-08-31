from ultralytics import YOLO
import torch

MODEL_PATH = "detect/train/weights/best.pt"
VIDEO_PATH = "videos/541331586_24293921923610669_3072870960600716785_n.mp4"
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
