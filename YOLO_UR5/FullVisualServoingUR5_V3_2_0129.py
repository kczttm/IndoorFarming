import socket
import time
import cv2
import numpy as np
import time
from yolov5.detect import yolo_detection
import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
# from SerialCommWithArduino import grip, ungrip

YOLO_WEIGHT_FILE = "./yolov5/runs/train/exp13/weights/best.pt"
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
        if msg == 'UR3_is_asking_for_data':
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

def get_berry_boxes():
    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)
    cam.set(4, 720)
    img = None
    final_boxes = None
    for i in range(10):
        result, img = cam.read()
        cv2.imwrite("blackberry_img_buffer.jpg",img)
        boxes = yolo_detection("blackberry_img_buffer.jpg", YOLO_WEIGHT_FILE)
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


# boxes = my_detection("new_image_1.png",YOLO_WEIGHT_FILE)
# frame = cv2.imread("new_image_1.png")
# for line in boxes:
#   cv2.rectangle(frame, (int(line[0]),int(line[1])), (int(line[2]),int(line[3])), (0,255,0), 10)
# cv2.imwrite("new_image_with_box.jpg",frame)


cap = cv2.VideoCapture("test_video.mp4")
 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
count = 0
 
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame_width,frame_height))
 
while(True):
  ret, frame = cap.read()
  count +=1
 
  if ret == True:
    cv2.imwrite("buffer.jpg",frame)
    boxes = yolo_detection("buffer.jpg",YOLO_WEIGHT_FILE)
    for line in boxes:
        cv2.rectangle(frame, (int(line[0]),int(line[1])), (int(line[2]),int(line[3])), (0,255,0), 8)
    out.write(frame)
 
  # Break the loop
  else:
    break 
 
# When everything done, release the video capture and video write objects
cap.release()
out.release()

# def estimate_depth(this_box):
#     this_box = this_box[0,0:4]
#     width = this_box[2] - this_box[0]
#     height = this_box[3] - this_box[1]
#     length = (width + height) / 2
#     corners = np.zeros((4,2))
#     corners[0,:] = np.array([this_box[0] - length/2,this_box[1] - length/2])
#     corners[1,:] = np.array([this_box[0] + length/2,this_box[1] - length/2])
#     corners[2,:] = np.array([this_box[0] + length/2,this_box[1] + length/2])
#     corners[3,:] = np.array([this_box[0] - length/2,this_box[1] + length/2])
#     temp = np.zeros((1,4,2))
#     temp[0] = corners
#     rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(temp,17/1000,cameraMatrix,dist)
#     depth_estimate = tvecs[0,0,2]
#     return depth_estimate

# def get_leftmost_centroid(centroids):
#     sorted_ccentroids = centroids[centroids[:, 0].argsort()]
#     return sorted_ccentroids[0,:]

# def change_brightness_and_contrast(img, brightness=255, contrast=127):
#     brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
 
#     contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
 
#     if brightness != 0:
 
#         if brightness > 0:
 
#             shadow = brightness
 
#             max = 255
 
#         else:
 
#             shadow = 0
#             max = 255 + brightness
 
#         al_pha = (max - shadow) / 255
#         ga_mma = shadow
 
#         # The function addWeighted
#         # calculates the weighted sum
#         # of two arrays
#         cal = cv2.addWeighted(img, al_pha,
#                               img, 0, ga_mma)
 
#     else:
#         cal = img
#     if contrast != 0:
#         Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
#         Gamma = 127 * (1 - Alpha)
 
#         # The function addWeighted calculates
#         # the weighted sum of two arrays
#         cal = cv2.addWeighted(cal, Alpha,
#                               cal, 0, Gamma)
 
#     return cal



# print("Starting Python program:")

# iteration = 0
# # Calibration to Find Image Jacobian Matrix
# while iteration < 5:
#     print("Calibration Iteration: ", iteration)

#     boxes = get_berry_boxes()
    
#     if boxes is None:
#         print("Could not detect Blackberry")
#         print("Trying again")
#     else:
#         centroids = get_centroids_from_boxes(boxes)
#         curr_centroid = get_leftmost_centroid(centroids)
#         tagCentroidX = curr_centroid[0]
#         tagCentroidY = curr_centroid[1]

#         if iteration == 0:
#             initObjCenter = [tagCentroidX, tagCentroidY]
#             print("P0: ", initObjCenter)
#             perturbX = 0.005
#             perturbY = 0
#             adjustArmXMeters = perturbX
#             adjustArmYMeters = perturbY

#         elif iteration == 1:
#             objCenter_PerturbX = [tagCentroidX, tagCentroidY]
#             print("P1: ", objCenter_PerturbX)
#             adjustArmXMeters = perturbX * -1 # Return to initial pose
#             adjustArmYMeters = 0

#         elif iteration == 2:
#             perturbX = 0
#             perturbY = 0.005       
#             adjustArmXMeters = perturbX
#             adjustArmYMeters = perturbY
        
#         elif iteration == 3:
#             objCenter_PerturbY = [tagCentroidX, tagCentroidY]
#             print("P2: ", objCenter_PerturbY)
#             adjustArmXMeters = 0
#             adjustArmYMeters = perturbY * -1 # Return to initial pose

#         elif iteration == 4:
#             adjustArmXMeters = 0
#             adjustArmYMeters = 0

#             # Image Jacobian matrix calculation (Reference Dr. Hu Slides for details)
#             a_11 = (objCenter_PerturbX[0] - initObjCenter[0]) / 0.01
#             a_21 = (objCenter_PerturbX[1] - initObjCenter[1]) / 0.01
#             a_12 = (objCenter_PerturbY[0] - initObjCenter[0]) / 0.01
#             a_22 = (objCenter_PerturbY[1] - initObjCenter[1]) / 0.01

#             imageJacobian = np.array([[a_11, a_12], [a_21, a_22]])
#             jPrime = np.linalg.inv(imageJacobian)
#             J_PRIME = jPrime
#         iteration += 1
#         move_robot([adjustArmXMeters, adjustArmYMeters, 0, 0, 0, 0])
#         time.sleep(0.3)
            

# print("Calibrated Image Jacobian Matrix: ", imageJacobian)
# print("Calibrated Inverse Image Jacobian Matrix: ", jPrime)

# xPrev = 0
# yPrev = 0  

# iteration = 1

# print()
# print("STARTING VISUAL SERVOING ADJUSTMENTS")
# print("************************************")
# print("************************************")
# print("************************************")
# print("************************************")
# print()

# curr_trip = np.zeros((6,))


# boxes = get_berry_boxes()
# print("\n")
# print("************************************")
# num_berries = (get_centroids_from_boxes(boxes)).shape[0]
# print("NUMBER OF BERRIES", num_berries)
# print("************************************")
# print("\n")

# for i in range(num_berries):
#     print("\n")
#     print("BERRY #: ", i+1)
#     print("\n")
#     adjustArmXY = True
#     while adjustArmXY is True:
#         while boxes is None:
#             print("Could not detect Blackberry")
#             boxes = get_berry_boxes()
#         curr_centroid = get_leftmost_centroid(get_centroids_from_boxes(boxes))
#         imError = (curr_centroid - IMAGE_CENTER).reshape(-1,1)
#         deltaChange = 1/3* (J_PRIME @ imError)
#         newAdjustments = (-deltaChange)
#         if newAdjustments[0][0] < 2.5e-3 and newAdjustments[1][0] < 2.5e-3:
#                 adjustArmXY = False
#         move_robot([newAdjustments[0][0], newAdjustments[1][0], 0, 0, 0, 0])
#         curr_trip += np.array([newAdjustments[0][0], newAdjustments[1][0], 0, 0, 0, 0])
#         time.sleep(0.3)
#         boxes = get_berry_boxes()

#     print()
#     print("************************************")
#     print("************************************")
#     print("XY Adjustment Complete")
#     print("************************************")
#     print("************************************")
#     print()

#     adjustZ = True
#     while adjustZ:
#         boxes = get_berry_boxes()
#         if boxes is None:
#             break
#         else:
#             all_centroids = get_centroids_from_boxes(boxes)
#             curr_centroid = get_leftmost_centroid(all_centroids)
#             leftmost_box = boxes[all_centroids[:, 0].argsort()]
#             print(leftmost_box)
#             depth = estimate_depth(leftmost_box)
#             print("depth: ", depth)
#             move_robot([0,0,depth*0.4,0,0,0])
#             curr_trip += np.array([0,0,depth*0.4,0,0,0])
#             if depth < 0.035:
#                 break

#             # Re-centering
#             adjustArmXY = True
#             while adjustArmXY is True:
#                 if depth < 0.035:
#                     break
#                 print("=====RE-CENTERING=======")
#                 for i in range(2):
#                     boxes = get_berry_boxes()
#                 if boxes is None:
#                     adjustZ = False
#                     break
#                 all_centroids = get_centroids_from_boxes(boxes)
#                 leftmost_box = boxes[all_centroids[:, 0].argsort()]
#                 depth = estimate_depth(leftmost_box)
#                 curr_centroid = get_leftmost_centroid(all_centroids)
#                 imError = (curr_centroid - IMAGE_CENTER).reshape(-1,1)
#                 deltaChange = 1/5* (J_PRIME @ imError)
#                 newAdjustments = (-deltaChange)
#                 if newAdjustments[0][0] < 2.5e-3 and newAdjustments[1][0] < 2.5e-3:
#                         adjustArmXY = False
#                         print("=====RE-CENTERING COMPLETE=======")
#                 move_robot([newAdjustments[0][0], newAdjustments[1][0], 0, 0, 0, 0])
#                 curr_trip += np.array([newAdjustments[0][0], newAdjustments[1][0], 0, 0, 0, 0])
#                 time.sleep(0.3)
            
#             time.sleep(0.3)
    
#     move_robot([0,0,20/1000,0,0,0])
#     curr_trip += np.array([0,0,20/1000,0,0,0])
#     print("Gripping begins: ")
#     print("+++++++++++++++++++++")
#     time.sleep(0.3)
#     grip()
#     move_robot([0,0,(-curr_trip)[2],0,0,0])
#     move_robot([(-curr_trip)[0],0,0,0,0,0])
#     move_robot([0,(-curr_trip)[1],0,0,0,0])
#     move_robot([0,0,0,np.pi/6,0,0])
#     ungrip()
#     move_robot([0,0,0,-np.pi/6,0,0])
#     time.sleep(0.3)
#     curr_trip = np.zeros((6,))


# print("Adjustments are done.")
# print("Ending program.")