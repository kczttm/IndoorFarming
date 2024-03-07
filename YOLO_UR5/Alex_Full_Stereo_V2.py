import sys
sys.path.append('../') # To call files from parent directory 
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
importTimerFinish = time.perf_counter()
print("\nTotal Time to Import Related Dependencies: " + str(importTimerFinish - importTimerStart) + "seconds \n")

## This script is for using stereo imaging depth estimation and does NOT use visual servoing. See "CompliantGripperIBVS_Yolo.py" for IBVS implementation of berry harvesting

# YOLO_WEIGHT_FILE = "./yolov5/runs/train/exp13/weights/best.pt"
YOLO_WEIGHT_FILE = "/Users/alex/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Documents/Georgia_Tech/GTRI FarmHand/Code/IndoorFarming/YOLO_UR5/yolov5/runs/train/exp13/weights/best.pt"

IMAGE_CENTER = [640, 360]

# path = "C:\\Users\\dkumar75\\Documents\\BerryDetectionImages"
max_intensity_threshold = 180
min_intensity_threshold = 100

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
THICKNESS = 2

# Color 
RED = (0, 0, 255) # OPENCV uses BGR not RGB
WHITE = (255, 255, 255)
MODEL = None

#### METHODS #####
def yolo_detection(source, weights):
    device = "cpu"
    imgsz = (640, 640)

    device = select_device(device)
    global MODEL
    if MODEL is None:
        # MODEL = DetectMultiBackend(weights, device=device, dnn=False, data="./yolov5/data/data.yaml", fp16=False)
        # Path for Macbook Pro 14 inch under Indoor Farming directory
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

def apply_yolo_to_image(img):
    final_boxes = None
    while final_boxes is None:
        print("in while")
        boxes = yolo_detection(img, YOLO_WEIGHT_FILE)

        if boxes is not None:
            frame = cv2.imread(img)
            for line in boxes:
                cv2.rectangle(frame, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0,255,0), 3)
                cv2.imwrite("apply_yolo_to_image_output.jpg", frame)
            final_boxes = boxes
        else:
            print("Did not detect berries, trying again.")

        centroids = get_centroids_from_boxes(final_boxes)

        for i in range(centroids.shape[0]):
            centroid = centroids[i]
            x = int(centroid[0])
            y = int(centroid[1])
            image = cv2.imread("apply_yolo_to_image_output.jpg")
            cv2.putText(image, str(centroid), (x, y - 80), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
            cv2.imwrite("apply_yolo_to_image_output.jpg", image)

        return frame, centroids

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

def change_brightness_and_contrast(img, brightness=255, contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
 
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
 
    if brightness != 0:
 
        if brightness > 0:
 
            shadow = brightness
 
            max = 255
 
        else:
 
            shadow = 0
            max = 255 + brightness
 
        al_pha = (max - shadow) / 255
        ga_mma = shadow
 
        # The function addWeighted
        # calculates the weighted sum
        # of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)
 
    else:
        cal = img
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
 
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)
 
    return cal

def main():

    # Initializing camera once to reduce inconsistency of camera initialization
    cam = cv2.VideoCapture(1)

    # Variables
    img_count = 1
    baseline = 0.005
    # tilt = np.radians(1.6339)
    tilt = np.radians(1.5)
    focal_length = 891.77161
    cam_x_offset = -0.017
    cam_y_offset = 0.000
    pitch_angle = np.radians(0) # Angle of pitch (radians) if camera isn't aligned parallel to ground and drifts up or down when moving forward (z-direction)
    stereoX = baseline
    stereoY = 0

    print("****************")
    print("STARTING PROGRAM")
    print("****************")

    iteration = 1

    # Figure out how many berries to determine loop condition
    print("\nTaking initialization image.")
    init_boxes, init_centroids, initialImageTime, yoloDetectionTimeInitial = get_berry_boxes_and_centroids("Initialization_Image.jpg", cam)

    print("\nInitial Image Time: ", initialImageTime)
    print("\nYolo Detection Time for Initial Image: ", yoloDetectionTimeInitial)

    num_berries_init = init_centroids.shape[0]
    print("\nINITIAL NUMBER OF R4 BERRIES DETECTED: " + str(num_berries_init))
    print("\nCentroids: ", init_centroids)
    totalR4 = num_berries_init

    num_berries = num_berries_init
    berries_left = num_berries_init

    print("\nENTERING LOOP ------------------------")

    # Start loop for harvesting
    while (num_berries != 0):

        cycleStart = time.perf_counter()

        print("\n\n\nStarting Stereo Calibration iteration: %2d" %iteration)

        # First stereo image 
        boxes1, centroids1, firstStereoImageTime, yoloDetectionTimeFirst = get_berry_boxes_and_centroids("AAA_First_Stereo_Image.jpg", cam)

        print("\nFirst Stereo Image Time: ", firstStereoImageTime)
        print("\nYolo Detection Time for First Stereo Image (included in above time): ", yoloDetectionTimeFirst)

        # print("\nCentroids 1: \n")
        # print(centroids1)
        centroids1 = centroids1[centroids1[:, 0].argsort()]
        # print("\nCentroids 1 sorted: \n")
        # print(centroids1)

        print("\nMoving robot to horizontal baseline position.")
        t1 = time.perf_counter()

        move_robot([stereoX, stereoY, 0, 0, 0, 0])

        t2 = time.perf_counter()
        print("Initial Horizontal Baseline Shift Move Command Time: ", (t2 - t1))
        time.sleep(1)

        # Take second stereo image
        print("\nTaking second stereo calibration image.")

        boxes2, centroids2, secondStereoImageTime, yoloDetectionTimeSecond = get_berry_boxes_and_centroids("AAA_Second_Stereo_Image.jpg", cam)

        print("\nSecond Stereo Image Time: ", secondStereoImageTime)
        print("\nYolo Detection Time for Second Stereo Image (included in above time): ", yoloDetectionTimeSecond)

        # print("\nCentroids 2: \n")
        # print(centroids2)

        centroids2 = centroids2[centroids2[:, 0].argsort()]
        # print("\nCentroids 2 sorted: \n")
        # print(centroids2)

        adjustments = np.zeros((centroids1.shape[0], 1, 3)) # Matrix of stored adjustment values for each berry location

        if centroids1.shape[0] != centroids2.shape[0]:
            print("\nDid not detect same number of berries in 2nd image, exiting program")
            exit()

        print("\nCentroids Matrix Shape: ", centroids1.shape[0])

        adjustmentCalculationTimeStart = time.perf_counter()

        for i in range(centroids1.shape[0]):

            print("\nIteration = ", i)

            # Get x1, y1 from first image
            firstCentroid = centroids1[i]
            # print("\nFirst Centroid: ", firstCentroid)
            x1, y1 = firstCentroid[0], firstCentroid[1]

            # Get x2, y2 from second image
            secondCentroid = centroids2[i]
            # print("\nSecond Centroid: ", secondCentroid)
            x2, y2 = secondCentroid[0], secondCentroid[1]
        
            # XYZ CALCULATIONS:

            xDisparity = np.abs(x2 - x1)
            # print("\nX Disparity = ", xDisparity)
            yDisparity = np.abs(y2 - y1)
            # print("\nY Disparity = ", yDisparity)
            worldImageConversion = baseline * np.cos(tilt) / xDisparity

            # z calculation uses disparity in x and y and averages in case they give different results. With horizontal baseline, there shouldn't be disparity in y
            # zMOVE = (((baseline*focal_length) / xDisparity) + ((baseline*focal_length) / yDisparity)) / 2 
            zMOVE = ((baseline*focal_length) / xDisparity)

            xOffsets = cam_x_offset - stereoX
            # print("\nX offsets = ", xOffsets)
            xError = worldImageConversion * (IMAGE_CENTER[0] - x1)
            # print("\nX error = ", xError)
            xMOVE = xOffsets - xError
            # print("\nX adjustment = (", xOffsets, ") - (", xError, ") = ", xMOVE)
            ## Image center is offset by camera offset from center of gripper 
            ## StereoX is baseline translation between 1st and 2nd image, should

            yOffsets = cam_y_offset - stereoY
            # print("\nY offsets = ", yOffsets)
            pitchOffset = -zMOVE * np.sin(pitch_angle)
            # print("\nPitch offset = ", pitchOffset)
            yError = -worldImageConversion * (IMAGE_CENTER[1] - y1) # Negated because camera is flipped upside down to match camera axes
            # print("\nY error = ", yError)
            yMOVE = pitchOffset + yOffsets + yError
            # print("\nY Adjustment = (", pitchOffset, ") + (", yOffsets, ") + (", yError, ") = ", yMOVE)
            adjustments[i][0][0], adjustments[i][0][1], adjustments[i][0][2] = xMOVE, yMOVE, zMOVE

        adjustmentCalculationTimeEnd = time.perf_counter()
        print("All Berries Adjustment Calculations Time: ", (adjustmentCalculationTimeEnd - adjustmentCalculationTimeStart))

        print("\nCalculated offsets for each berry: \n")
        print(adjustments)

        if iteration == 3:
            xMOVE = adjustments[0][0][0]
            yMOVE = adjustments[0][0][1] - 0.003
            zMOVE = adjustments[0][0][2]
        elif iteration == 2:
            xMOVE = adjustments[0][0][0]
            yMOVE = adjustments[0][0][1] - 0.003
            zMOVE = adjustments[0][0][2]
        else:
            xMOVE = adjustments[0][0][0]
            yMOVE = adjustments[0][0][1] - 0.003
            zMOVE = adjustments[0][0][2]

        # Move gripper back to 1st stereo position
        print("\nMoving back to 1st stereo imaging home position.")
        t1 = time.perf_counter()
        move_robot([-stereoX, -stereoY, 0, 0, 0, 0])
        t2 = time.perf_counter()

        print("Move Arm Back to Home Position Before Approach Time: ", (t2 - t1))
            
        # CENTERING MOVEMENT WITH CAMERA OFFSET

        print("\nX offset: " + str(xMOVE) + ". Y offset: " + str(yMOVE) + ". Z offset: " + str(zMOVE))
        
        t1 = time.perf_counter()
        time.sleep(1)
        initialXCommand = xMOVE + stereoX
        initialYCommand = yMOVE + stereoY
        initialZCommand = zMOVE
        move_robot([initialXCommand, initialYCommand, 0, 0, 0, 0])
        move_robot([0, 0, initialZCommand - 0.041, 0, 0, -np.pi/2])
        t2 = time.perf_counter()
        print("\nInitial Approach Time: ", (t2 - t1))

        time.sleep(0.75)
        t1 = time.perf_counter()
        com.startGrip() # Communicate with arduino to initiate half grip
        time.sleep(1)
        move_robot([0, 0, 0.04 - 0.04, 0, 0, 0])
        t2 = time.perf_counter()

        print("\nStart Grip and Move Robot Forward in Z Time: ", (t2 - t1))

        t1 = time.perf_counter()

        time.sleep(2)
        com.finishGrip()
        time.sleep(0.75)

        t2 = time.perf_counter()
        
        print("\nFinish Gripping Time: ", (t2 - t1))

        returnHomeStart = time.perf_counter()

        # Return to home position and drop blackberries
        move_robot([0, 0, -initialZCommand, 0, 0, 0])
        time.sleep(1)
        com.startGrip()
        move_robot([0, 0, 0, 0, 0, np.pi/2])
        move_robot([-initialXCommand, -initialYCommand, 0, 0, 0, 0])
        time.sleep(0.5)
        com.ungrip()

        returnHomeEnd = time.perf_counter()

        print("\nReturn to Home Position and Drop Berry Time: ", (returnHomeEnd - returnHomeStart))

        cycleEnd = time.perf_counter()
        print("\nTotal Approach + Harvesting Cycle Time (not including initial berry scan): " + str(cycleEnd - cycleStart))
        time.sleep(1.5)

        num_berries -= 1
        iteration += 1

    print("\nProgram finished.")
    totalTimerEnd = time.perf_counter()
    print("\nTotal Time Taken: ", (totalTimerEnd - totalTimerStart))

if __name__ == "__main__":
    main()