import cv2
import numpy as np
import mediapipe as mp
from math import atan2, pi

# =========================
# Config
# =========================
N_RADIAL = 64       # จำนวน sampling ตามแนวรัศมี
N_ANGULAR = 256     # จำนวน sampling ตามแนวเชิงมุม (0..2pi)
GABOR_KSIZE = 9     # ขนาด kernel Gabor
GABOR_SIGMA = 2.0
GABOR_LAMBDA = 8.0
GABOR_GAMMA = 0.5
GABOR_THETAS = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # มุมตัวกรอง
THRESH_BIN = 0.0   # เกณฑ์ binarize (0 = แยกบวก/ลบ)
# หมายเหตุ: ปรับ N_ANGULAR สูงขึ้นจะละเอียดขึ้นแต่ช้าขึ้น

# =========================
# MediaPipe FaceMesh (iris landmarks)
# =========================
mp_face_mesh = mp.solutions.face_mesh

# iris indices (ตาม MediaPipe)
RIGHT_IRIS = [468, 469, 470, 471, 472]  # โดยทั่วไป 468 ~ center
LEFT_IRIS  = [473, 474, 475, 476, 477]  # โดยทั่วไป 473 ~ center

def detect_landmarks_bgr(img_bgr):
    """คืนค่า multi_face_landmarks (list) หรือ None"""
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,            # ต้องเปิดเพื่อได้ iris
        min_detection_confidence=0.5
    ) as fm:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = fm.process(rgb)
    return res.multi_face_landmarks if res.multi_face_landmarks else None

def to_px(lmk, shape):
    h, w = shape[:2]
    return int(lmk.x * w), int(lmk.y * h)

def extract_eye_params(landmarks, eye_indices, shape):
    """
    ประมาณ center และ radius ของ iris แบบง่าย:
      - center = จุด index แรก (468 หรือ 473)
      - radius = ค่าเฉลี่ยระยะจาก center ไปยังอีก 4 จุด
    คืนค่า (center(x,y), r_pupil, r_iris)
    เราใช้ r_pupil ~ 0.5*radius, r_iris ~ radius (คร่าวๆ)
    """
    pts = [to_px(landmarks[i], shape) for i in eye_indices]
    center = pts[0]  # 468 หรือ 473
    dists = [np.hypot(p[0]-center[0], p[1]-center[1]) for p in pts[1:]]
    if len(dists) == 0:
        return None
    r_iris = float(np.mean(dists))
    r_pupil = max(1.0, 0.5 * r_iris)  # ค่าประมาณหยาบ
    return center, r_pupil, r_iris

# =========================
# Rubber Sheet Normalization
# =========================
def rubber_sheet(gray, center, r_pupil, r_iris, n_radial=N_RADIAL, n_angular=N_ANGULAR):
    """
    คลี่วงแหวนจาก r_pupil -> r_iris เป็นภาพขนาด (n_radial x n_angular)
    พิกัดเชิงมุม θ เดิน 0..2π, รัศมีเดินเชิงเส้น
    คืนค่า normalized, mask (บริเวณที่ออกนอกภาพจะเป็น 0 ใน mask)
    """
    h, w = gray.shape[:2]
    cx, cy = center
    thetas = np.linspace(0, 2*np.pi, n_angular, endpoint=False)
    radii  = np.linspace(r_pupil, r_iris, n_radial)

    # target grid
    xs = np.zeros((n_radial, n_angular), dtype=np.float32)
    ys = np.zeros((n_radial, n_angular), dtype=np.float32)

    for a_idx, th in enumerate(thetas):
        cos_t, sin_t = np.cos(th), np.sin(th)
        xs[:, a_idx] = cx + radii * cos_t
        ys[:, a_idx] = cy + radii * sin_t

    # sample ด้วย remap
    map_x = xs
    map_y = ys
    normalized = cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # mask: 1 เฉพาะพิกเซลที่อยู่ในภาพเดิม
    mask = ((map_x >= 0) & (map_x < w) & (map_y >= 0) & (map_y < h)).astype(np.uint8)
    return normalized, mask

# =========================
# Enhancement + Gabor Feature → Iris Code
# =========================
def enhance(gray_norm):
    """
    ปรับ contrast แบบเบา ๆ ด้วย CLAHE
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = clahe.apply(gray_norm)
    return enh

def gabor_kernels(ksize=GABOR_KSIZE, sigma=GABOR_SIGMA, lambd=GABOR_LAMBDA, gamma=GABOR_GAMMA, thetas=GABOR_THETAS):
    kernels = []
    for th in thetas:
        k = cv2.getGaborKernel((ksize, ksize), sigma, th, lambd, gamma, 0, ktype=cv2.CV_32F)
        # zero-mean เพื่อให้บิตแบ่งบวก/ลบสมดุลขึ้น
        k -= k.mean()
        kernels.append(k)
    return kernels

def iris_code_from_norm(norm_img):
    """
    สร้าง iris code แบบง่าย:
      - apply Gabor filters หลายมุม
      - รวม feature (sum หรือ concat)
      - binarize: > THRESH_BIN = 1, else 0
    คืนค่า (code_uint8, mask_valid)
    """
    enh = enhance(norm_img)
    kernels = gabor_kernels()
    feats = []
    for k in kernels:
        resp = cv2.filter2D(enh, cv2.CV_32F, k)
        feats.append(resp)
    feat_sum = np.sum(feats, axis=0)

    code = (feat_sum > THRESH_BIN).astype(np.uint8)  # 0/1
    # mask: พื้นที่ที่เป็นศูนย์ล้วน ๆ (จากขอบ/นอกภาพ) อาจทำให้ค่าผิดเพี้ยน
    # ประมาณง่าย ๆ: พิกเซลที่เท่ากับ 0 ทั้งก่อนและหลัง enhance ให้เป็น invalid
    mask_valid = (norm_img > 0).astype(np.uint8)
    return code, mask_valid

# =========================
# Matching (Hamming Distance with mask)
# =========================
def hamming_distance(code1, mask1, code2, mask2):
    """
    คำนวณ Hamming distance โดยใช้เฉพาะตำแหน่งที่ทั้งสองฝั่ง valid
    """
    valid = (mask1.astype(bool) & mask2.astype(bool))
    if np.count_nonzero(valid) == 0:
        return 1.0  # เทียบไม่ได้
    xor = np.logical_xor(code1[valid], code2[valid]).astype(np.uint8)
    return xor.mean()

# =========================
# High-level pipeline
# =========================
def get_iris_code_from_image(img_bgr, which_eye="left"):
    """
    which_eye: 'left' หรือ 'right'
    คืนค่า (code, mask) หรือ (None, None) ถ้า fail
    """
    lms_all = detect_landmarks_bgr(img_bgr)
    if not lms_all:
        return None, None
    lms = lms_all[0].landmark
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if which_eye == "left":
        params = extract_eye_params(lms, LEFT_IRIS, img_bgr.shape)
    else:
        params = extract_eye_params(lms, RIGHT_IRIS, img_bgr.shape)
    if params is None:
        return None, None

    center, r_pupil, r_iris = params

    # Rubber sheet
    norm, mask = rubber_sheet(gray, center, r_pupil, r_iris, N_RADIAL, N_ANGULAR)
    # สร้าง iris code
    code, valid = iris_code_from_norm(norm)
    # รวม mask (remap-valid * feature-valid)
    final_mask = (mask & valid).astype(np.uint8)

    return code, final_mask

# =========================
# Enrollment / Verification demo (1:1)
# =========================
def enroll_person(images_bgr, which_eye="left"):
    """
    รับภาพหลายภาพของคนเดียว -> รวม iris code (majority vote)
    """
    codes = []
    masks = []
    for img in images_bgr:
        c, m = get_iris_code_from_image(img, which_eye=which_eye)
        if c is not None:
            codes.append(c.astype(np.int8))
            masks.append(m.astype(np.uint8))
    if len(codes) == 0:
        return None, None
    # รวม mask: valid ถ้า valid ในอย่างน้อยครึ่งหนึ่งของภาพ
    stack_mask = np.stack(masks, axis=0)
    agg_mask = (np.mean(stack_mask, axis=0) > 0.5).astype(np.uint8)
    # majority vote ของบิต
    stack_codes = np.stack(codes, axis=0)
    vote = (np.mean(stack_codes, axis=0) > 0.5).astype(np.uint8)
    # ถ้า mask จุดไหนไม่ valid ให้บิตเป็น 0 (ไม่ใช้ตอนเทียบอยู่แล้ว)
    vote[agg_mask == 0] = 0
    return vote, agg_mask

def verify(query_img_bgr, enrolled_code, enrolled_mask, which_eye="left"):
    qcode, qmask = get_iris_code_from_image(query_img_bgr, which_eye=which_eye)
    if qcode is None:
        return None
    dist = hamming_distance(qcode, qmask, enrolled_code, enrolled_mask)
    return dist

# =========================
# Quick demo (แก้ path แล้วรันท้ายไฟล์)
# =========================
if __name__ == "__main__":
    # --- ตั้ง path ตัวอย่าง ---
    # enrollment: ภาพของ "บุคคล A" หลายใบ
    enroll_paths = [
        "data/enroll/IMG_0020.JPEG",
        "data/enroll/IMG_0021.JPEG",
        "data/enroll/IMG_0022.JPEG",
        "data/enroll/IMG_0023.JPEG",
        "data/enroll/IMG_0024.JPEG",
        "data/enroll/IMG_0025.JPEG",
        "data/enroll/IMG_0026.JPEG",
        "data/enroll/IMG_0027.JPEG",
        "data/enroll/IMG_0028.JPEG",
        "data/enroll/IMG_0029.JPEG"
    ]
    # query: ภาพที่อยากเช็คว่าใช่ A ไหม
    query_path = "data/verify/IMG_5645.JPEG"
    which_eye = "left"  # หรือ "right"

    # โหลดภาพ
    enroll_imgs = []
    for p in enroll_paths:
        img = cv2.imread(p)
        if img is not None:
            enroll_imgs.append(img)

    if len(enroll_imgs) == 0:
        print("ไม่พบภาพ enrollment ลองแก้ path ก่อนครับ")
        exit(0)

    enrolled_code, enrolled_mask = enroll_person(enroll_imgs, which_eye=which_eye)
    if enrolled_code is None:
        print("Enrollment ล้มเหลว (หา iris ไม่เจอ/คุณภาพไม่ดี)")
        exit(0)

    query_img = cv2.imread(query_path)
    if query_img is None:
        print("ไม่พบภาพ query ลองแก้ path ก่อนครับ")
        exit(0)

    dist = verify(query_img, enrolled_code, enrolled_mask, which_eye=which_eye)
    if dist is None:
        print("ตรวจ query ไม่สำเร็จ")
        exit(0)

    # เกณฑ์เบื้องต้น (ต้องปรับตามข้อมูลจริงของคุณ)
    # ยิ่ง "ต่ำ" ยิ่งคล้าย: ปกติถ้าเป็น IR แท้ ๆ เราจะตั้ง ~0.3 หรือต่ำกว่านั้น
    # สำหรับ RGB ทดลอง อาจเห็นค่า 0.35–0.45 ยังพอรับได้ (ต้องลองวัดบนชุดข้อมูลของคุณ)
    THRESH = 0.40

    print(f"Hamming distance = {dist:.4f}")
    if dist < THRESH:
        print("=> ถือว่า 'ใช่คนเดิม' (PASS)")
    else:
        print("=> 'ไม่น่าใช่คนเดิม' (FAIL)")