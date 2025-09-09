import cv2
import torch
from ultralytics import YOLO
import os

# -----------------------------
# ตั้งค่า
# -----------------------------
MODEL_PATH = "detect/train/weights/best.pt"  # ที่อยู่ไฟล์โมเดล
CONF_THRES = 0.25
CAMERA_DEVICE = "/dev/video0"                # เปลี่ยนให้ตรงกับ Iriun ของคุณ

# สเกลการแสดงผล (ย่อเฉพาะตอนโชว์ ไม่กระทบที่ส่งเข้าโมเดล)
DISPLAY_SCALE = 0.6     # 0.6 = ย่อเหลือ 60% (ปรับ 0.3-0.8 ได้ตามชอบ)

# ขนาดหน้าต่างบังคับ (หากอยากกำหนดแน่นอนแทนการสเกล)
FORCE_WINDOW_SIZE = (960, 540)  # กว้าง x สูง พิกเซล (ปรับได้)

# -----------------------------
# โหลดโมเดล
# -----------------------------
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)

# -----------------------------
# เปิดกล้อง Iriun บน Ubuntu
# -----------------------------
if not os.path.exists(CAMERA_DEVICE):
    raise RuntimeError(f"ไม่พบอุปกรณ์กล้อง: {CAMERA_DEVICE}")

cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("ไม่สามารถเปิดกล้อง Iriun ได้")

# ตั้งค่าความละเอียด (ต้นฉบับที่ให้โมเดลตรวจจับ)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# -----------------------------
# ตั้งค่าหน้าต่างแสดงผล
# -----------------------------
WIN_NAME = "YOLO Face Detection (Iriun)"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)     # ให้ยืด/ย่อได้
cv2.resizeWindow(WIN_NAME, *FORCE_WINDOW_SIZE)   # บังคับขนาดเริ่มต้น
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
# หมายเหตุ: ถ้าใช้หน้าจอ HiDPI แล้วหน้าต่างยังใหญ่ ลองลด FORCE_WINDOW_SIZE หรือ DISPLAY_SCALE ลงอีก

print("กด [+]/[-] ซูมเข้า/ออก, f = เต็มจอ/ปกติ, r = รีเซ็ตขนาด, q/ESC = ออก")

# -----------------------------
# วนลูปอ่านภาพ
# -----------------------------
# สถานะโหมดเต็มจอ
is_fullscreen = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านภาพจากกล้องได้")
        break

    # ส่งเฟรมเข้า YOLO
    results = model.predict(
        source=frame,
        conf=CONF_THRES,
        device=device,
        verbose=False
    )

    # วาดผลลัพธ์บนเฟรม (เฟรมขนาดต้นฉบับ)
    annotated = results[0].plot()

    # -------- วิธีที่ 1: ย่อเฉพาะตอนแสดงผล --------
    if DISPLAY_SCALE != 1.0:
        display_w = int(annotated.shape[1] * DISPLAY_SCALE)
        display_h = int(annotated.shape[0] * DISPLAY_SCALE)
        annotated_display = cv2.resize(annotated, (display_w, display_h), interpolation=cv2.INTER_AREA)
    else:
        annotated_display = annotated

    # แสดงผล
    cv2.imshow(WIN_NAME, annotated_display)

    # คีย์ลัดควบคุมหน้าต่าง
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    elif key == ord("+") or key == ord("="):     # ซูมเข้า
        DISPLAY_SCALE = min(DISPLAY_SCALE + 0.05, 1.0)
    elif key == ord("-") or key == ord("_"):     # ซูมออก
        DISPLAY_SCALE = max(DISPLAY_SCALE - 0.05, 0.2)
    elif key == ord("f"):                        # toggle fullscreen
        is_fullscreen = not is_fullscreen
        cv2.setWindowProperty(
            WIN_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
        )
        if not is_fullscreen:
            # กลับสู่โหมดปกติแล้วปรับขนาดหน้าต่างให้พอดี
            cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(WIN_NAME, *FORCE_WINDOW_SIZE)
    elif key == ord("r"):                        # รีเซ็ตขนาด
        DISPLAY_SCALE = 0.6
        if is_fullscreen:
            is_fullscreen = False
            cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, *FORCE_WINDOW_SIZE)

cap.release()
cv2.destroyAllWindows()
