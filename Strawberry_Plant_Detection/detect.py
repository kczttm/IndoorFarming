from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

def detect(input_image_name, confidence=0.9):

    # Absolute path for model weights on Mac 14 pro 
    model = YOLO("/Users/alex/Documents/Georgia Tech/GTRI FarmHand/Code/IndoorFarming/Strawberry_Plant_Detection/runs/detect/train20/weights/best.pt")
    results = model.predict(source=input_image_name, conf=confidence)

    num_instances = len(results[0])
    print("Number of instances detected: ", num_instances)

    # Process each result
    for result in results:

        boxes = result.boxes

        classes = result.names
        print("Classes: ", classes)

        labels = boxes.cls.numpy()
        # print("Labels: ", labels)

        num_flowers = np.count_nonzero(labels == 0)
        num_stamen = np.count_nonzero(labels == 1)

        print("Number of flowers detected: ", num_flowers)
        print("Number of stamen detected: ", num_stamen)

        # Check if there are any detections and boxes.xyxy is not empty
        if boxes is not None and len(boxes.xyxy) > 0:

            frame = cv2.imread(input_image_name) # load image 

            # Iterate through each detected box
            for ind, (box, confidence) in enumerate(zip(boxes.xyxy, boxes.conf)):

                x1, y1, x2, y2 = map(int, box) # x1, y1 are top-left corner; x2, y2 are bottom-right corner
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) # draw green rectangle
                cv2.putText(frame, f'Confidence: {confidence:.2f}', (x1 - 130, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # annotate image
                
                if labels[ind] == 0:
                    box_label = "Flower"
                else:
                    box_label = "Stamen"

                cv2.putText(frame, box_label, (x1 - 130, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Save the annotated image after drawing all boxes
            cv2.imwrite('detect_output.png', frame)

        else:
            print("No flower or stamen detected!")

    return frame, boxes, num_flowers, num_stamen