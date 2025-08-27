import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import torch


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO Eye-ID realtime on webcam")
    ap.add_argument("--weights", default="best.pt", help="path to best.pt")
    ap.add_argument("--source", default="0", help="camera index (e.g., 0) หรือไฟล์วิดีโอ/RTSP")
    ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size")
    ap.add_argument("--device", default=None, help="cuda, cuda:0, หรือ cpu (ค่าปริยาย: auto)")
    ap.add_argument("--width", type=int, default=1280, help="ความกว้างของหน้าต่างแสดงผล (resize เฉพาะการแสดงผล)")
    ap.add_argument("--show_fps", action="store_true", help="แสดง FPS")
    return ap.parse_args()


def auto_device(user_device):
    if user_device:
        return user_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def draw_label(img, text, x, y):
    # กล่องพื้นหลังให้ข้อความอ่านง่าย
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - h - baseline - 4), (x + w + 6, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def color_for_class(cls_id):
    # ทำสีคงที่ตาม class id
    np.random.seed(cls_id + 7)
    c = np.random.randint(0, 255, size=3).tolist()
    return (int(c[0]), int(c[1]), int(c[2]))


def main():
    args = parse_args()
    device = auto_device(args.device)

    # โหลดโมเดล
    model = YOLO(args.weights)
    model.to(device)

    # เปิด source (เว็บแคมหรือวิดีโอ)
    src = args.source
    if src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"ไม่สามารถเปิด source: {args.source}")

    names = model.names  # dict: id -> class name (ควรเป็นชื่อบุคคลในชุดข้อมูล)
    prev_t = time.time()
    fps = 0.0

    win_name = "YOLO Eye-ID (press 'q' to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # inference
        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

        # วาดกรอบ/ป้าย
        if results.boxes is not None:
            for box in results.boxes:
                # พิกัดกรอบ
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                label_name = names.get(cls_id, f"id:{cls_id}")
                label = f"{label_name} {conf:.2f}"

                color = color_for_class(cls_id if cls_id >= 0 else 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, label, x1, y1)

        # FPS
        if args.show_fps:
            now = time.time()
            dt = now - prev_t
            prev_t = now
            fps = (0.9 * fps + 0.1 * (1.0 / dt)) if fps > 0 else (1.0 / dt)
            draw_label(frame, f"FPS: {fps:.1f}", 10, 30)

        # resize เพื่อการแสดงผล (ไม่กระทบ inference)
        h, w = frame.shape[:2]
        if w != args.width and args.width > 0:
            new_h = int(h * (args.width / w))
            show_frame = cv2.resize(frame, (args.width, new_h))
        else:
            show_frame = frame

        cv2.imshow(win_name, show_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
