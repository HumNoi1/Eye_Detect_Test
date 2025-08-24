from ultralytics import YOLO

model = YOLO('best.pt')  # load a pretrained YOLOv11n model

results = model('video/222f2a10-9815-4bb0-9a25-ed56fb3e26bf.mp4')  # predict on an image

results[0].show()


# video