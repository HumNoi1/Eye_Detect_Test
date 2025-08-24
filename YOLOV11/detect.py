from ultralytics import YOLO

model = YOLO('best.pt')  # load a pretrained YOLOv11n model

results = model('test/FullSizeRender-4_jpg.rf.6a0e598a6b0e6d5326d24efccd8e12ca.jpg')  # predict on an image

results[0].show()