from ultralytics import YOLO
import ultralytics
ultralytics.checks()
from roboflow import Roboflow

# rf = Roboflow(api_key="ttCrOerdRSQWJH6lroQp")
# project = rf.workspace("alex-qiu-ml").project("strawberry-flower-detection")
# version = project.version(1)
# dataset = version.download("yolov8")
# print(dataset)
# print(dir(dataset))

# Create new model using custom data.yaml file
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
results = model.train(data="original_res_endoscope_artificial_flowers.yolov8/data.yaml", epochs=25, device='mps') # model saved to runs/detect folder under weights --> "best.pt"

# Validate model
metrics = model.val(data="original_res_endoscope_artificial_flowers.yolov8/data.yaml")

# Test model

for bounding_box in detections:
    x1 = bounding_box['x'] - bounding_box['width'] / 2
    x2 = bounding_box['x'] + bounding_box['width'] / 2
    y1 = bounding_box['y'] - bounding_box['height'] / 2
    y2 = bounding_box['y'] + bounding_box['height'] / 2
    box = (x1, x2, y1, y2)

# Export model to onnx format 
model.export(format='onnx', dynamic=True)


