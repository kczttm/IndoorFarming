import socket
import time
import cv2
import numpy as np
import time
from yolov5.detect import yolo_detection
import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from .utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from .utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from .utils.plots import Annotator, colors, save_one_box
from .utils.torch_utils import select_device, time_sync
# from SerialCommWithArduino import grip, ungrip

if torch.cuda.is_available():
    mac = False
    curr_device = 'cuda'
    print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    # Windows path 
    YOLO_WEIGHT_FILE = "C:\\Users\\Alex\\OneDrive - Georgia Institute of Technology\\Documents\\Georgia_Tech\\GTRI FarmHand\\Code\\IndoorFarming\\YOLO_UR5\\best.pt"
else:
    curr_device = 'cpu'
    mac = True
    # Macbook path
    YOLO_WEIGHT_FILE = "/Users/alex/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Documents/Georgia_Tech/GTRI FarmHand/Code/IndoorFarming/YOLO_UR5/yolov5/runs/train/exp13/weights/best.pt"

cameraMatrix = np.array([[927.31957258, 0,667.19142084],[0,922.20248778,335.69393703],[0,0,1]])
dist = np.array([[-0.17574952,0.65288341, -0.00300312,  0.00724758, -0.95447869]])
IMAGE_CENTER = np.array([640, 360])
J_PRIME = np.eye(2)
MODEL = None

def yolo_detection(source, weights):
    device="cpu"
    imgsz=(640, 640)

    device = select_device(device)
    global MODEL
    if MODEL is None:

        if mac is not True:
            # Windows path?
            MODEL = DetectMultiBackend(weights, device=device, dnn=False, data="C:\\Users\\Alex\\OneDrive - Georgia Institute of Technology\\Documents\\Georgia_Tech\\GTRI FarmHand\\Code\\IndoorFarming\\YOLO_UR5\\data.yaml", fp16=False)
        else:
            # Macbook path
            MODEL = DetectMultiBackend(weights, device=device, dnn=False, data="/Users/alex/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Documents/Georgia_Tech/GTRI FarmHand/Code/IndoorFarming/YOLO_UR5/data.yaml", fp16=False)
    
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
    HOST = "169.254.14.134"  # Laptop (server) IP address.
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

def get_berry_boxes():
    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)
    cam.set(4, 720)
    img = None
    final_boxes = None
    for i in range(10):
        result, img = cam.read()
        cv2.imwrite("hikaru.jpg",img)
        boxes = yolo_detection("hikaru.jpg", YOLO_WEIGHT_FILE)
        if boxes is None:
            continue
        elif boxes.shape[0] > 3:
            continue
        elif final_boxes is None:
            final_boxes = boxes
        elif boxes.shape[0] >= final_boxes.shape[0]:
            final_boxes = boxes
            if final_boxes.shape[0] == 3:
                break
    print(final_boxes)
    return final_boxes

def apply_yolo_to_image(image):

    boxes = yolo_detection(image,YOLO_WEIGHT_FILE)
    centroids = get_centroids_from_boxes(boxes)
    # print(boxes)
    frame = cv2.imread(image)
    for line in boxes:
        cv2.rectangle(frame, (int(line[0]),int(line[1])), (int(line[2]),int(line[3])), (0,255,0), 10)
        cv2.imwrite("Test_YOLO_Output.jpg",frame)

    return frame, centroids

# a = apply_yolo_to_image("Berry_Leaf_1_2.5mm_offset.png")
# print("Successfully Detected Berry!")
