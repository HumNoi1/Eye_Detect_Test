import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# สไตล์การวาด
draw_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def get_landmarks(image_bgr):
    """
    รับภาพ BGR (จาก cv2.imread) -> คืนค่า (result, landmarks) ของ MediaPipe
    - แปลง BGR เป็น RGB ก่อนประมวลผล
    - ป้องกัน None ถ้าไม่พบหน้า
    """
    # MediaPipe ต้องการ RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ใช้ context manager เพื่อปิดทรัพยากรอัตโนมัติ
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,              # เปิด iris (จะมีจุด ~478 จุด)
        min_detection_confidence=0.5
    ) as face_mesh:
        # set writeable=False เพื่อเร่งความเร็ว
        image_rgb.flags.writeable = False
        result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        return result, None

    landmarks = result.multi_face_landmarks[0].landmark
    return result, landmarks

def draw_landmarks(image_bgr, result):
    """วาด tesselation, contours และ irises ลงบนภาพ BGR"""
    if not result or not result.multi_face_landmarks:
        return image_bgr

    annotated = image_bgr.copy()
    for face_landmark in result.multi_face_landmarks:
        # ตาข่าย
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmark,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        # เส้นขอบใบหน้า/ปาก/ตา
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmark,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        # ม่านตา
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmark,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=draw_specs,
            connection_drawing_spec=draw_specs,
        )
    return annotated

def to_pixel_coords(landmark, image_shape):
    """
    แปลง normalized landmark (x,y) -> พิกเซล (int)
    landmark.x, landmark.y อยู่ในช่วง [0,1]
    """
    h, w = image_shape[:2]
    x_px = int(landmark.x * w)
    y_px = int(landmark.y * h)
    return x_px, y_px

def get_iris_centers(landmarks, image_shape):
    """
    คืนพิกัดศูนย์กลางม่านตาซ้าย/ขวาแบบง่าย ๆ
    ใช้จุดสำคัญจาก FACEMESH_IRISES:
      - ตาขวา (Right iris) indices: 468–472 (โดยปกติ 468 = center)
      - ตาซ้าย (Left iris)  indices: 473–477 (โดยปกติ 473 = center)
    """
    if landmarks is None:
        return None, None

    # ป้องกัน index error: เมื่อ refine_landmarks=True จะมี ~478 จุด
    if len(landmarks) < 478:
        return None, None

    right_center = to_pixel_coords(landmarks[468], image_shape)
    left_center  = to_pixel_coords(landmarks[473], image_shape)
    return left_center, right_center

if __name__ == "__main__":
    path_img = "data/IMG_0854.JPEG"
    img = cv2.imread(path_img)

    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพที่: {path_img}")

    result, landmarks = get_landmarks(img)
    annotated = draw_landmarks(img, result)

    # ตัวอย่าง: คำนวณจุดศูนย์กลางม่านตา แล้ววาดวงกลมเล็ก ๆ ไว้ดู
    left_iris, right_iris = get_iris_centers(landmarks, img.shape)
    if left_iris:
        cv2.circle(annotated, left_iris, 3, (0, 255, 0), -1)
    if right_iris:
        cv2.circle(annotated, right_iris, 3, (0, 255, 0), -1)

    # แสดงภาพ
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original Image", 800, 600)
    cv2.namedWindow("Annotated Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotated Image", 800, 600)

    cv2.imshow("Original Image", img)
    cv2.imshow("Annotated Image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
