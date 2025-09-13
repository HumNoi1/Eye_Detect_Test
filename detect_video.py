from ultralytics import YOLO
import torch

MODEL_PATH = "detect/train/weights/best.pt"
VIDEO_PATH = "videos/543678357_24738357705794822_5252272891705357099_n.mp4"
# เลือกอุปกรณ์อัตโนมัติ (ถ้ามี CUDA ก็ใช้ GPU)
device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO(MODEL_PATH)

# แสดงผลแบบสดบนจอ (ไม่บันทึกไฟล์) — ไม่มีการแตะต้อง rotation ใดๆ
model.predict(
    source=VIDEO_PATH,
    conf=0.5,
    device=device,
    show=True,        # เปิดหน้าต่าง imshow
    verbose=True
)
