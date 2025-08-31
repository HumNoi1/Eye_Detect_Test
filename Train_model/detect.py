from ultralytics import YOLO

# โหลดโมเดล
model = YOLO('detect/train/weights/best.pt')

# predict ทั้งโฟลเดอร์ แล้วเซฟผลใน runs/detect/test/
results = model.predict(
    source="dataset/test/images",   # โฟลเดอร์ภาพที่ต้องการทดสอบ
    save=True,                      # บันทึกผลเป็นไฟล์ภาพ
    project="Test/predict",          # โฟลเดอร์หลัก
    name="test"                     # โฟลเดอร์ย่อย (จะได้ runs/detect/test/)
)