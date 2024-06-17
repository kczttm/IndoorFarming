import cv2
import torch
from torchvision import transforms
from torchvision.models import detection
from PIL import Image
import os, sys

# Get the absolute path of the current script
repo_root = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
sys.path.append(repo_root)
from Strawberry_Plant_Detection.detect import detect

# # Load the pre-trained YOLOv3 model
# model = detection.yolo_v3(pretrained=True)

# # Set the model to evaluation mode
# model.eval()

# # Load the COCO class labels
# with open("coco_labels.txt", "r") as f:
#     labels = f.read().splitlines()

# # Define the transformation to apply to the input image
# transform = transforms.Compose([
#     transforms.Resize((416, 416)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# Initialize the video capture
cap = cv2.VideoCapture(2,cv2.CAP_V4L2)

if not cap.isOpened():
        print("Cannot open camera")
        sys.exit()

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()
    if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    
    # stricktly not converting!!
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB because openCV uses BGR

    # yolov8 prediction
    annotated_image, boxes, num_flowers, num_stamen = detect(frame, confidence=0.7)
    # Display the frame
    cv2.imshow("Object Detection", annotated_image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()