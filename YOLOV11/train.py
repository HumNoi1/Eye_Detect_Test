from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # load a pretrained YOLOv11n model

# Train the model
train_results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    device="0",
    workers=8,
    cache=True,
    batch=-1,  # auto batch size
    project="YOLOV11",
    name=""
)

metrics = model.val()