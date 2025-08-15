# eye_crop.py
#เทรน/ลงทะเบียน/ทดสอบ

import cv2, numpy as np, mediapipe as mp

mp_mesh = mp.solutions.face_mesh

def _landmarks_to_np(landmarks, w, h, idxs):
    pts = []
    for i in idxs:
        lm = landmarks.landmark[i]
        pts.append((int(lm.x * w), int(lm.y * h)))
    return np.array(pts, dtype=np.int32)

# ดัชนีรอบตา (MediaPipe Face Mesh)
LEFT_EYE_IDX  = list(range(33, 133))   # ครอบคลุมรอบตาซ้าย (เผื่อเหลือ)
RIGHT_EYE_IDX = list(range(263, 463))  # ครอบคลุมรอบตาขวา

def crop_eyes_both(img_bgr, margin=0.25, target=112):
    """คืนลิสต์ [eye_left, eye_right] (BGR -> Gray 112x112) ถ้าหาไม่เจอ คืน []"""
    h, w = img_bgr.shape[:2]
    with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks: 
            return []

        lms = res.multi_face_landmarks[0]
        # หา bounding box รอบตาซ้าย/ขวา
        le_pts = _landmarks_to_np(lms, w, h, LEFT_EYE_IDX)
        re_pts = _landmarks_to_np(lms, w, h, RIGHT_EYE_IDX)

        eyes = []
        for pts in [le_pts, re_pts]:
            x,y,w0,h0 = cv2.boundingRect(pts)
            cx, cy = x + w0/2, y + h0/2
            side = int(max(w0, h0) * (1+margin))
            x1 = max(0, int(cx - side/2)); y1 = max(0, int(cy - side/2))
            x2 = min(w, int(cx + side/2));  y2 = min(h, int(cy + side/2))
            eye = img_bgr[y1:y2, x1:x2]
            if eye.size == 0: 
                continue
            gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            eye128 = cv2.resize(gray, (target, target))
            eyes.append(eye128)
        return eyes
