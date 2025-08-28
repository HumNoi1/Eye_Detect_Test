from ultralytics import YOLO

model = YOLO('yolov9t.pt')  # load a pretrained YOLOv9t model

# Train the model
train_results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    device="0",
    workers=2,
    cache=True,
    batch=-1,  # auto batch size
    project="runs/detect",
    name="train"
)

metrics = model.val()
print(metrics)