import time
importTimerStart = time.perf_counter()
totalTimerStart = time.perf_counter()
import socket
import cv2
import numpy as np
from yolov5.detect import yolo_detection
import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                            increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
# import SerialCommWithArduino as com
import rrt
importTimerFinish = time.perf_counter()
print("\nTotal Time to Import Related Dependencies: " + str(importTimerFinish - importTimerStart) + "seconds \n")

#### METHODS #####
def yolo_detection(source, weights):
    device = "cpu"
    imgsz = (640, 640)

    device = select_device(device)
    global MODEL
    if MODEL is None:
        MODEL = DetectMultiBackend(weights, device=device, dnn=False, data="./yolov5/data/data.yaml", fp16=False)
    model = MODEL
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = False
        pred = model(im, augment=False, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, 0.7,0.45, None, False, 1000)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                return det.numpy()

def stereo_base(x, y, quad_1_x, quad_1_y):
    if (x <= 640 and y <= 360):
        print("quadrant IV")
        return quad_1_x, -quad_1_y
    elif (x >= 640 and y <= 360):
        print("quadrant I")
        return  -quad_1_x, -quad_1_y
    elif (x >= 640 and y >= 360):
        print("quadrant II")
        return  -quad_1_x, quad_1_y
    elif (x <= 640 and y >= 360):
        print("quadrant III")
        return quad_1_x, quad_1_y
    else:
        0.007, 0.007

def get_centroids_from_boxes(boxes):
    centroids = np.zeros((boxes.shape[0], 2))
    centroids[:,0] = (boxes[:,0] + boxes[:,2]) / 2
    centroids[:,1] = (boxes[:,1] + boxes[:,3]) / 2
    return np.int64(centroids)

def build_pose_string(pose):
    # pose is a list with length of 6
    # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
    values = str(pose[0]) + ", " + str(pose[1]) + ", " + str(pose[2]) + \
        ", " + str(pose[3]) + ", " + str(pose[4]) + ", " + str(pose[5])
    poseString = "(" + values + ")"
    return poseString

def move_robot(pose):
    # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
    # HOST = "169.254.14.134"  # Laptop (server) IP address.
    HOST = "169.254.14.134"
    PORT = 30002

    # Communicate with UR5
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))  # Bind to the port.
    s.listen(5)  # Wait for UR5 (client) connection.
    c, addr = s.accept()  # Establish connection with client.

    try:
        msg = c.recv(1024).decode()  # Receive message from UR5.
        if msg == 'UR5_is_asking_for_data':
            poseString = build_pose_string(pose)
            c.send(poseString.encode())
            c.close()
            s.close()

    except socket.error as socketerror:
            print("Error occured.")

def get_centroids_from_boxes(boxes):
    centroids = np.zeros((boxes.shape[0], 2))
    centroids[:,0] = (boxes[:,0] + boxes[:,2]) / 2
    centroids[:,1] = (boxes[:,1] + boxes[:,3]) / 2
    return np.int64(centroids)

def get_berry_boxes_and_centroids(imageName, cam): 

    t1 = time.perf_counter()

    imgPath = "./OUTPUT_IMAGES"
    # cam = cv2.VideoCapture(0) #Change between 0 and 1 and 2 (sometimes change from 0 to 1 to -1 back to 0 sometimes works), check stack overflow if acting funny again b/c its very inconsistent)
    cam.set(3, 1280)
    cam.set(4, 720)
    img = None
    final_boxes = None
    while final_boxes is None:
        result, img = cam.read()
        cv2.imwrite("BerryImage.jpg", img)
        t3 = time.perf_counter()
        boxes = yolo_detection("BerryImage.jpg", YOLO_WEIGHT_FILE)
        t4 = time.perf_counter()
        yoloDuration = t4 - t3

        # Saving an annotated image to see which berries were detected
        if boxes is not None:
            frame = cv2.imread("BerryImage.jpg")
            for line in boxes:
                cv2.rectangle(frame, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0,255,0), 3)
                cv2.imwrite(imageName, frame)
            final_boxes = boxes
        
        else:
            print("Did not detect berries, trying again.")

    centroids = get_centroids_from_boxes(final_boxes)
    
    for i in range(centroids.shape[0]):
        centroid = centroids[i]
        x = int(centroid[0])
        y = int(centroid[1])
        image = cv2.imread(imageName)
        cv2.putText(image, str(centroid), (x, y - 80), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
        cv2.imwrite(imageName, image)

    t2 = time.perf_counter()
    methodDuration = t2 - t1

    return final_boxes, centroids, methodDuration, yoloDuration

def draw_label(im, label, x, y):
    """Draw text onto image at location."""

    # Text parameters.
    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    THICKNESS = 1

    # Color 
    RED = (255, 0, 0)

    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED)
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

def estimate_depth(this_box):
    this_box = this_box[0,0:4]
    width = this_box[2] - this_box[0]
    height = this_box[3] - this_box[1]
    length = (width + height) / 2
    corners = np.zeros((4,2))
    corners[0,:] = np.array([this_box[0] - length/2,this_box[1] - length/2])
    corners[1,:] = np.array([this_box[0] + length/2,this_box[1] - length/2])
    corners[2,:] = np.array([this_box[0] + length/2,this_box[1] + length/2])
    corners[3,:] = np.array([this_box[0] - length/2,this_box[1] + length/2])
    temp = np.zeros((1,4,2))
    temp[0] = corners
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(temp,17/1000,cameraMatrix,dist)
    depth_estimate = tvecs[0,0,2]
    return depth_estimate

def get_leftmost_centroid(centroids):
    if centroids.shape[0] == 1:
        print("\nOnly one centroid: ", centroids)
        return centroids[0]
    else:
        print("\nDetected %2d berries.", centroids.shape[0])
        sorted_centroids = centroids[centroids[:, 0].argsort()]
        print("\nSorted Centroids: ", sorted_centroids)
        return sorted_centroids[0]