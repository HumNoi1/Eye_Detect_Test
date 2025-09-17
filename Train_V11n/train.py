from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # load a pretrained YOLOv11n model

# Train the model
train_results = model.train(
    data='/home/humnoi1/Documents/dataset/data.yaml',  # path to your dataset directory
    epochs=100,
    imgsz=640,
    device="0",
    workers=8,
    cache=True,
    batch=24,  # adjust based on your GPU memory
)

metrics = model.val()