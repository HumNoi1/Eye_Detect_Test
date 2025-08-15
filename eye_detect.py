import cv2

# โหลดโมเดลตรวจจับใบหน้าและดวงตา
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)  # เปลี่ยนเป็น 1 ถ้ามีกล้องหลายตัว

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120,120))

    for (x, y, w, h) in faces:
        # วาดกรอบใบหน้า
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # ตรวจจับดวงตาเฉพาะในโซนใบหน้า
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=6, minSize=(24,24))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (255,0,0), 2)

    cv2.imshow('Eye Detection (Haar Cascade)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
