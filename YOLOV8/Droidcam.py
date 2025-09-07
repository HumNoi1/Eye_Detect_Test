import cv2
import torch
from ultralytics import YOLO
import os
import glob

# -----------------------------
# ตั้งค่า
# -----------------------------
MODEL_PATH = "detect/train/weights/best.pt"
CONF_THRES = 0.5

# ตัวช่วย: หาอุปกรณ์วิดีโอที่ชื่อสื่อถึง OBS/DroidCam/Iriun
PREFERRED_NAMES = ["OBS Virtual Camera", "DroidCam", "Iriun"]

DISPLAY_SCALE = 0.6
FORCE_WINDOW_SIZE = (960, 540)

# -----------------------------
# ฟังก์ชันค้นหาอุปกรณ์กล้อง v4l2 ตามชื่อการ์ด
# -----------------------------
def find_preferred_v4l2(names=PREFERRED_NAMES):
    # ดูรายชื่อทั้งหมด /dev/video*
    candidates = sorted(glob.glob("/dev/video*"))
    # อ่านชื่อการ์ดจาก /sys/class/video4linux/videoX/name
    for dev in candidates:
        base = os.path.basename(dev)             # เช่น "video2"
        sys_name_file = f"/sys/class/video4linux/{base}/name"
        try:
            with open(sys_name_file, "r", encoding="utf-8") as f:
                card_name = f.read().strip()
            for want in names:
                if want.lower() in card_name.lower():
                    return dev, card_name
        except Exception:
            continue
    # ถ้าไม่เจอชื่อที่ต้องการ ให้ลองตัวแรกที่เปิดได้
    for dev in candidates:
        if os.path.exists(dev):
            return dev, None
    return None, None

# -----------------------------
# โหลดโมเดล
# -----------------------------
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)

# -----------------------------
# เลือกอุปกรณ์กล้องอัตโนมัติ (เน้น OBS Virtual Camera/DroidCam)
# -----------------------------
CAMERA_DEVICE, CARD_NAME = find_preferred_v4l2()
if CAMERA_DEVICE is None:
    raise RuntimeError("ไม่พบอุปกรณ์ /dev/video* บนระบบ")

print(f"ใช้กล้อง: {CAMERA_DEVICE}" + (f" ({CARD_NAME})" if CARD_NAME else ""))

cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError(f"ไม่สามารถเปิดกล้องได้: {CAMERA_DEVICE}")

# ตั้งค่าความละเอียดต้นฉบับที่ส่งเข้าโมเดล (ปรับตามที่ OBS ส่งออก)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# -----------------------------
# ตั้งค่าหน้าต่างแสดงผล
# -----------------------------
WIN_NAME = "YOLO Detection (OBS/DroidCam on Fedora)"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, *FORCE_WINDOW_SIZE)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

print("กด [+]/[-] ซูมเข้า/ออก, f = เต็มจอ/ปกติ, r = รีเซ็ตขนาด, q/ESC = ออก")

is_fullscreen = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านภาพจากกล้องได้ (ตรวจว่า OBS กด Start Virtual Camera แล้วหรือยัง)")
        break

    # inference
    results = model.predict(
        source=frame,
        conf=CONF_THRES,
        device=device,
        verbose=False
    )

    annotated = results[0].plot()

    # ย่อเฉพาะตอนแสดงผล
    if DISPLAY_SCALE != 1.0:
        display_w = int(annotated.shape[1] * DISPLAY_SCALE)
        display_h = int(annotated.shape[0] * DISPLAY_SCALE)
        annotated_display = cv2.resize(annotated, (display_w, display_h), interpolation=cv2.INTER_AREA)
    else:
        annotated_display = annotated

    cv2.imshow(WIN_NAME, annotated_display)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    elif key in (ord("+"), ord("=")):
        DISPLAY_SCALE = min(DISPLAY_SCALE + 0.05, 1.0)
    elif key in (ord("-"), ord("_")):
        DISPLAY_SCALE = max(DISPLAY_SCALE - 0.05, 0.2)
    elif key == ord("f"):
        is_fullscreen = not is_fullscreen
        cv2.setWindowProperty(
            WIN_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
        )
        if not is_fullscreen:
            cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(WIN_NAME, *FORCE_WINDOW_SIZE)
    elif key == ord("r"):
        DISPLAY_SCALE = 0.6
        if is_fullscreen:
            is_fullscreen = False
            cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, *FORCE_WINDOW_SIZE)

cap.release()
cv2.destroyAllWindows()
