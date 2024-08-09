import cv2
import numpy as np
import os
import torch
import copy

# Eabling Yolo to detect larger image with smaller objects
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction

repo_root = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
# Absolute path for model weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sahi_detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=repo_root+"/Strawberry_Plant_Detection/runs/detect/train9/weights/best.pt",
    device=device,
    confidence_threshold=0.5,
)


def detect_sahi(frame, slice_height=256, slice_width=256, overlap_h_ratio=0.2, overlap_w_ratio=0.2):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pred_results = get_sliced_prediction(
        detection_model=sahi_detection_model,
        image=frame,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_h_ratio,
        overlap_width_ratio=overlap_w_ratio,
        perform_standard_pred=False,
        verbose=0,
    )

    # # Perform standard prediction using yolov8
    # pred_results = get_prediction(
    #     detection_model=sahi_detection_model,
    #     image=frame,
    #     verbose=1,
    # )

    pred_list = pred_results.object_prediction_list
    return pred_list


def draw_sahi_boxes(frame, pred_list, depth_img, CameraInfo, H_wd_rs=None):
    # if H_wd_rs is not None:
    # output flower centers in world frame
    # otherwise, output flower centers in camera frame

    color=(0, 255, 0)
    # rect_th=max(round(sum(frame.shape) / 2 * 0.003), 2)
    # text_th=max(rect_th-1, 1)
    # text_size=rect_th/3
    rect_th, text_th, text_size = 2, 2, 0.75

    flower_centers = []

    # print(rect_th, text_th, text_size)

    # Process each result
    # add bboxes to image if present
    for object_prediction in pred_list:
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()

        bbox = object_prediction.bbox.to_xyxy()
        category_name = object_prediction.category.name
        if category_name != "flower":
            continue
        score = object_prediction.score.value

        # set bbox points
        point1, point2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        # get the depth value at the center of the bounding box
        center_x = (point1[0] + point2[0]) // 2
        center_y = (point1[1] + point2[1]) // 2
        [x, y, z] = pixels_to_meters(center_x, center_y, depth_img, CameraInfo)
        if H_wd_rs is not None:
            # convert flower center to world frame
            flower_center = np.array([x, y, z, 1])
            flower_center = np.dot(H_wd_rs, flower_center)
            x, y, z = flower_center[0], flower_center[1], flower_center[2]
        flower_centers.append([x, y, z])
            
        # visualize boxes
        cv2.rectangle(
            frame,
            point1,
            point2,
            color=color,
            thickness=3,
        )

        # arange bounding box text location
        # add depth to the label
        label = f"[{x:.3f}, {y:.3f}, {z:.3f}]"
        # label += f" {score:.2f}"

        box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]  # label width, height
        outside = point1[1] - box_height - 3 >= 0  # label fits outside box
        point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
        # add bounding box text
        cv2.rectangle(frame, point1, point2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            frame,
            label,
            (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
            0,
            text_size,
            (0, 0, 0),
            thickness=text_th,
        )
    return frame, np.array(flower_centers)


def pixels_to_meters(pixel_x, pixel_y, depth_img, CameraInfo):
    # Get the depth value at the pixel location
    depth = depth_img[pixel_y, pixel_x] / 1000.0  # convert to meters

    # Get the intrinsics of the camera
    # note that the camera is rotated 90 degrees, so the x and y are swapped
    fy = CameraInfo.k[0]
    fx = CameraInfo.k[4]
    cy = CameraInfo.k[2]
    cx = CameraInfo.k[5]

    # Calculate the x and y distance from the center of the image
    x = (pixel_x - cx) * depth / fx
    y = (pixel_y - cy) * depth / fy

    return x, y, depth
