from ultralytics import YOLO

model = YOLO('best.pt')  # load a pretrained YOLOv11n model

results = model('picture/9647e0a8-ff91-4776-a776-a09b496b0364.jpeg')  # predict on an image

results[0].show()