import os, time, argparse, pickle
import cv2
import numpy as np
from sklearn.preprocessing import normalize

import tensorflow as tf
# ทำให้ TF จอง VRAM แบบเพิ่มทีละน้อย (ช่วยเวลาใช้งานร่วมกับ OpenCV/mediapipe)
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

import mediapipe as mp
from keras_facenet import FaceNet

# ---------- Utils ----------
def prewhiten(x: np.ndarray) -> np.ndarray:
    mean, std = np.mean(x), np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    return (x - mean) / std_adj

def crop_face_bgr(frame_bgr, bbox, margin=0.3, target_size=(160,160)):
    h, w = frame_bgr.shape[:2]
    x, y, ww, hh = bbox
    cx, cy = x + ww/2, y + hh/2
    m = margin
    new_w, new_h = ww*(1+m), hh*(1+m)
    x1 = int(max(0, cx - new_w/2))
    y1 = int(max(0, cy - new_h/2))
    x2 = int(min(w, cx + new_w/2))
    y2 = int(min(h, cy + new_h/2))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, target_size)
    crop_rgb = prewhiten(crop_rgb.astype('float32'))
    return crop_rgb

def l2_distance(a, b):
    return np.linalg.norm(a - b)

# ---------- Detection with MediaPipe ----------
class MPFaceDetector:
    def __init__(self, min_score=0.6, model_selection=0):
        self.mp = mp.solutions
        self.detector = self.mp.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_score
        )

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.detector.process(frame_rgb)
        boxes = []
        if res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x, y = int(bb.xmin * w), int(bb.ymin * h)
                ww, hh = int(bb.width * w), int(bb.height * h)
                x, y = max(0, x), max(0, y)
                ww, hh = max(1, ww), max(1, hh)
                conf = d.score[0] if d.score else 0.0
                boxes.append(((x, y, ww, hh), conf))
        return boxes

# ---------- Embedding (FaceNet) ----------
class FaceEmbedder:
    def __init__(self):
        self.model = FaceNet()
    def embed(self, face_rgb_160):
        # FaceNet จาก keras-facenet รับ list ของภาพ RGB (160x160x3) แบบ float
        emb = self.model.embeddings([face_rgb_160])[0]
        # ทำ L2 normalize ไว้จะจับคู่ด้วย cosine/L2 ก็ได้
        emb = normalize(emb.reshape(1, -1))[0]
        return emb

# ---------- Simple DB ----------
def load_db(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}  # {name: [emb, emb, ...]}

def save_db(db, path):
    with open(path, 'wb') as f:
        pickle.dump(db, f)

def add_sample(db, name, emb):
    db.setdefault(name, [])
    db[name].append(emb)

def best_match(db, emb, metric='l2', thresh=1.1):
    best_name, best_score = None, 1e9 if metric=='l2' else -1e9
    for name, vecs in db.items():
        # ใช้ค่า min distance ของชื่อเดียวกัน
        if metric == 'l2':
            d = min(l2_distance(emb, v) for v in vecs)
            if d < best_score:
                best_score, best_name = d, name
        else:
            # cosine similarity
            cs = max(float(np.dot(emb, v)) for v in vecs)
            if cs > best_score:
                best_score, best_name = cs, name
    if metric == 'l2':
        return (best_name if best_score <= thresh else "Unknown", best_score)
    else:
        # กรณี cosine เลือก threshold ~0.5-0.7 ตามคุณภาพข้อมูล
        return (best_name if best_score >= 0.6 else "Unknown", best_score)

# ---------- Main Loop ----------
def run_enroll(name, db_path, shots=20, cam=0):
    cap = cv2.VideoCapture(cam)
    detector = MPFaceDetector()
    embedder = FaceEmbedder()
    db = load_db(db_path)

    taken = 0
    last_t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break
        boxes = detector.detect(frame)
        if boxes:
            # เอาหน้าใหญ่สุด
            (bx,by,bw,bh),conf = max(boxes, key=lambda x: x[0][2]*x[0][3])
            face_rgb = crop_face_bgr(frame, (bx,by,bw,bh))
            if face_rgb is not None:
                emb = embedder.embed(face_rgb)
                add_sample(db, name, emb)
                taken += 1
                cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (0,255,0), 2)
                cv2.putText(frame, f"Enrolling {name}: {taken}/{shots}", (bx, by-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        # แสดง FPS
        now = time.time(); fps = 1/(now-last_t) if now>last_t else 0; last_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Enroll", frame)
        if taken >= shots:
            print(f"[INFO] Collected {shots} embeddings for {name}.")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    save_db(db, db_path)
    cap.release(); cv2.destroyAllWindows()
    print(f"[OK] Saved DB to {db_path}")

def run_recognize(db_path, metric='l2', thresh=1.1, cam=0):
    db = load_db(db_path)
    if not db:
        print(f"[WARN] DB is empty. Run enroll first.")
    cap = cv2.VideoCapture(cam)
    detector = MPFaceDetector()
    embedder = FaceEmbedder()

    last_t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break
        boxes = detector.detect(frame)
        # วาดทุกหน้าที่เจอ
        for (bx,by,bw,bh),conf in boxes:
            face_rgb = crop_face_bgr(frame, (bx,by,bw,bh))
            label, score = "Unknown", 0.0
            if face_rgb is not None and db:
                emb = embedder.embed(face_rgb)
                label, score = best_match(db, emb, metric=metric, thresh=thresh)
            color = (0,255,0) if label!="Unknown" else (0,0,255)
            cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), color, 2)
            cv2.putText(frame, f"{label}", (bx, by-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # FPS
        now = time.time(); fps = 1/(now-last_t) if now>last_t else 0; last_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Recognize", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["enroll","recognize"], required=True)
    ap.add_argument("--name", help="ชื่อคน (ตอน enroll)")
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--db", default="embeddings.pkl")
    ap.add_argument("--metric", choices=["l2","cosine"], default="l2")
    ap.add_argument("--thresh", type=float, default=1.1)
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "enroll":
        if not args.name:
            raise SystemExit("ต้องระบุ --name ตอน enroll")
        run_enroll(args.name, args.db, shots=args.shots, cam=args.cam)
    else:
        run_recognize(args.db, metric=args.metric, thresh=args.thresh, cam=args.cam)
