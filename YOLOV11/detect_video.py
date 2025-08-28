import cv2
import time
import subprocess
import json
import shutil
import torch
from ultralytics import YOLO

# ----------------------------
# ตั้งค่า
# ----------------------------
MODEL_PATH = "best.pt"
VIDEO_PATH = "videos/1d19f44f-fff5-44cd-b0d5-4d346648634e.mp4"
CONF_THRES = 0.25           # ความมั่นใจขั้นต่ำในการ detect
MAX_DISPLAY_WIDTH = 1280    # จำกัดความกว้างตอนแสดงผลเพื่อความลื่น (ตั้ง None ถ้าไม่อยากจำกัด)
SHOW_RAW = False            # True เพื่อดูภาพดิบก่อนวาด bbox

# ----------------------------
# Utilities: อ่าน rotation จาก metadata
# ----------------------------
def has_ffprobe():
    return shutil.which("ffprobe") is not None

def get_video_rotation_ffprobe(path: str) -> int | None:
    """
    คืนค่ามุมหมุนตาม metadata: 0/90/180/270 หรือ None ถ้าหาไม่ได้
    ต้องมี ffprobe ในเครื่อง
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=side_data_list:stream_tags=rotate",
            "-print_format", "json",
            path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8"))

        # 1) ลองจาก tags.rotate
        tags = data.get("streams", [{}])[0].get("tags", {})
        if "rotate" in tags:
            rot = int(tags["rotate"]) % 360
            return rot

        # 2) ลองจาก side_data_list (displaymatrix)
        sdl = data.get("streams", [{}])[0].get("side_data_list", [])
        for item in sdl:
            if "rotation" in item:
                rot = int(item["rotation"]) % 360
                return rot
    except Exception:
        pass
    return None

def rotate_frame(frame, rotation: int | None):
    """
    หมุนภาพตาม rotation (องศา) ถ้า None จะไม่หมุน
    """
    if rotation is None or rotation == 0:
        return frame
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # มุมอื่น ๆ ไม่คาดคิด: ข้ามไป
    return frame

def maybe_guess_rotation_by_shape(frame, expected_portrait=True) -> int | None:
    """
    เดามุมหมุนจากสัดส่วนภาพ หากคาดว่าเป็นวิดีโอแนวตั้ง (portrait)
    และเฟรมปัจจุบันเป็นแนวนอน → แนะนำหมุน 90 องศา
    """
    h, w = frame.shape[:2]
    is_portrait = h > w
    if expected_portrait and not is_portrait:
        # ส่วนใหญ่จากมือถือจะต้อง 90 CW
        return 90
    return None

# ----------------------------
# โหลดโมเดล + เลือกอุปกรณ์
# ----------------------------
model = YOLO(MODEL_PATH)
device = 0 if torch.cuda.is_available() else "cpu"
# หมายเหตุ: ใช้ device ใน predict แต่ถ้าอยากย้ายโมเดลไป device ตลอดก็ได้:
# model.to("cuda" if device == 0 else "cpu")

# ----------------------------
# เปิดวิดีโอ
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"เปิดวิดีโอไม่ได้: {VIDEO_PATH}")

# อ่าน rotation จาก metadata ถ้ามี
rotation = None
if has_ffprobe():
    rotation = get_video_rotation_ffprobe(VIDEO_PATH)

# ----------------------------
# Loop อ่านเฟรม + Detect
# ----------------------------
prev_t = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ถ้ายังไม่รู้ rotation และอยากเดาว่าไฟล์จากมือถือ (portrait)
    if rotation is None:
        rotation = maybe_guess_rotation_by_shape(frame, expected_portrait=True)

    # หมุนตาม metadata / เดา
    frame = rotate_frame(frame, rotation)

    # แสดงภาพดิบสำหรับ debug ถ้าต้องการ
    if SHOW_RAW:
        cv2.imshow("RAW Frame", frame)

    # (ทางเลือก) ลดขนาดก่อนแสดงผลเพื่อความลื่น
    if MAX_DISPLAY_WIDTH is not None:
        h, w = frame.shape[:2]
        if w > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # เรียก YOLO (ใช้ predict เพื่อกำหนด device/conf ได้ชัดเจน)
    # รองรับทั้ง BGR numpy frame โดยตรง
    results = model.predict(source=frame, conf=CONF_THRES, device=device, verbose=False)

    # วาดผลลัพธ์
    annotated = results[0].plot()

    # คำนวณ FPS ง่าย ๆ
    now = time.time()
    fps = 1.0 / max(now - prev_t, 1e-6)
    prev_t = now
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # แสดงผล
    cv2.imshow("YOLO Detection", annotated)

    # ปุ่มออก: q หรือ ESC
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
