from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # load a pretrained YOLOv11n model

# Train the model
train_results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    device="0",
    workers=0
)

metrics = model.val()