import cv2
import torch
from ultralytics import YOLO

# -----------------------------
# ตั้งค่า
# -----------------------------
MODEL_PATH = "best_YOLOV9m.pt"
CONF_THRES = 0.25

# โหลดโมเดล
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)

# เปิดกล้อง (0 = webcam หลัก)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("ไม่สามารถเปิดกล้องได้")

print("กด q เพื่อออก")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ส่งเฟรมเข้า YOLO
    results = model.predict(
        source=frame,
        conf=CONF_THRES,
        device=device,
        verbose=False
    )

    # วาดผลลัพธ์บนเฟรม
    annotated = results[0].plot()

    # แสดงผล
    cv2.imshow("YOLO Face Detection (Webcam)", annotated)

    # ออกด้วย q หรือ ESC
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
