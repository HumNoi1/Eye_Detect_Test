from ultralytics import YOLO

model = YOLO('YOLOV11/train/weights/best.pt')  # load a pretrained YOLO model

results = model('picture/ad78799e-3915-47a1-8cce-ceefaa42bbee.jpeg')  # predict on an image
#results = model('dataset/test/images/IMG_3431_jpg.rf.27d9c29998b0b8f7c71033a1a6755f63.jpg')

results[0].show()


#from ultralytics import YOLO

# โหลดโมเดล
#model = YOLO('detect/detect3/weights/best.pt')

# predict ทั้งโฟลเดอร์ แล้วเซฟผลใน runs/detect/test/
#results = model.predict(
    #source="dataset/test/images",   # โฟลเดอร์ภาพที่ต้องการทดสอบ
    #save=True,                      # บันทึกผลเป็นไฟล์ภาพ
    #project="Test/predict",          # โฟลเดอร์หลัก
    #name="test"                     # โฟลเดอร์ย่อย (จะได้ runs/detect/test/)
#)
