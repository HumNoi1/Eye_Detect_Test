from ultralytics import YOLO
import cv2

# โหลดโมเดล
model = YOLO("best.pt")

# เปิดไฟล์วิดีโอ
video_path = "video/ea43ab13-a66d-4cb8-aede-e1f43227e814.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect แต่ละ frame
    results = model(frame)

    # วาดผลลัพธ์ลงบน frame
    annotated_frame = results[0].plot()

    # แสดงผล
    cv2.imshow("YOLO Detection", annotated_frame)

    # กด q เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
