from ultralytics import YOLO

model = YOLO('best.pt')  # load a pretrained YOLOv11n model

results = model('picture/dc1ac1d3-e539-4158-9c4a-53ee9b889909.jpeg')  # predict on an image

results[0].show()