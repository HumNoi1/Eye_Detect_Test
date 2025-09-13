from ultralytics import YOLO

# โหลดโมเดล
model = YOLO('detect/train/weights/best.pt')

# predict ทั้งโฟลเดอร์ แล้วเซฟผลใน runs/detect/test/
results = model.predict(
    source="picture/538291289_1521331212199340_8227825501333211853_n.jpg",   # โฟลเดอร์ภาพที่ต้องการทดสอบ
)