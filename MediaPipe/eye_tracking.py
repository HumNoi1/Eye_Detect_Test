import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh # 468 landmarks
mp_drawing_styles = mp.solutions.drawing_styles
draw_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def get_landmarks(image):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                      refine_landmarks=True, min_detection_confidence=0.5)
    image.flags.writeable = False
    result = face_mesh.process(image)
    landmarks = result.multi_face_landmarks[0].landmark
    return result, landmarks

def draw_landmarks(image, result):
    image.flags.writeable = True
    if result.multi_face_landmarks:
        for face_landmark in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_list=face_landmark,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=draw_specs,
                connection_drawing_spec=draw_specs
            )
    return image

path_img = 'data/IMG_0854.JPEG'
img = cv2.imread(path_img)
annotated_img = img.copy()
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", 800, 600)
cv2.namedWindow("Annotated Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotated Image", 800, 600)

result, landmarks = get_landmarks(image=img)

annotated_img = draw_landmarks(image=annotated_img, result=result)
cv2.imshow("Original Image", img)
cv2.imshow("Annotated Image", annotated_img)

cv2.waitKey(0)
cv2.destroyAllWindows()