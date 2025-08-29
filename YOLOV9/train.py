from ultralytics import YOLO

model = YOLO('yolov11n.pt')  # load a pretrained YOLOv11n model

# Train the model
train_results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    device="0",
    workers=8,
    cache=True,
    batch=-1,  # auto batch size
)

metrics = model.val()