# Make sure to pip install ultralytics
# https://docs.ultralytics.com/datasets/classify/cifar10/
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='cifar10', epochs=60, imgsz=32)