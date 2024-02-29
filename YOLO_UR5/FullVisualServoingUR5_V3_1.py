import socket
import time
import cv2
import numpy as np
from cmath import pi
import time
from yolov5.detect import yolo_detection

YOLO_WEIGHT_FILE = "/Users/tonytu/Desktop/Soft_Robotics_Internship/YOLO_UR5_Test/yolov5/runs/train/exp10/weights/best.pt"
IMAGE_CENTER = np.array([640, 360])
J_PRIME = np.eye(2)

cameraMatrix = np.array([[927.31957258, 0,667.19142084],[0,922.20248778,335.69393703],[0,0,1]])
dist = np.array([[-0.17574952,0.65288341, -0.00300312,  0.00724758, -0.95447869]])

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
    for i in range(5):
        result, img = cam.read()
    cv2.imwrite("blackberry_img_buffer.jpg",img)
    boxes = yolo_detection("blackberry_img_buffer.jpg", YOLO_WEIGHT_FILE)
    return boxes

def center_robot_to(img_point):
    # img_point should be np.array of shape (2,)
    adjustArmXY = True
    while adjustArmXY is True:
        imError = (img_point - IMAGE_CENTER).reshape(-1,1)
        deltaChange = 1/3* (jPrime @ imError)
        newAdjustments = (-deltaChange)
        if newAdjustments[0][0] < 1e-4 and newAdjustments[1][0] < 1e-4:
            adjustArmXY = False
        move_robot([newAdjustments[0][0], newAdjustments[1][0], 0, 0, 0, 0])
        time.sleep(1.5)


trajectory = np.array([595.5,-44.02,375.12]).reshape(-1,1)
print("Starting Python program:")

adjustArm = True
iteration = 0

# Calibration to Find Image Jacobian Matrix
while iteration < 5:

    print("Calibration Iteration: ", iteration)
    
    boxes = get_berry_boxes()
    
    if boxes is None:
        print("Could not detect Blackberry")
        print("Trying again")
    else:
        centroids = get_centroids_from_boxes(boxes)
        curr_centroid = centroids[0,:]
        tagCentroidX = curr_centroid[0]
        tagCentroidY = curr_centroid[1]

        if iteration == 0:
            initObjCenter = [tagCentroidX, tagCentroidY]
            print("P0: ", initObjCenter)
            perturbX = 0.005
            perturbY = 0
            adjustArmXMeters = perturbX
            adjustArmYMeters = perturbY

        elif iteration == 1:
            objCenter_PerturbX = [tagCentroidX, tagCentroidY]
            print("P1: ", objCenter_PerturbX)
            adjustArmXMeters = perturbX * -1 # Return to initial pose
            adjustArmYMeters = 0

        elif iteration == 2:
            perturbX = 0
            perturbY = 0.005       
            adjustArmXMeters = perturbX
            adjustArmYMeters = perturbY
        
        elif iteration == 3:
            objCenter_PerturbY = [tagCentroidX, tagCentroidY]
            print("P2: ", objCenter_PerturbY)
            adjustArmXMeters = 0
            adjustArmYMeters = perturbY * -1 # Return to initial pose

        elif iteration == 4:
            adjustArmXMeters = 0
            adjustArmYMeters = 0

            # Image Jacobian matrix calculation (Reference Dr. Hu Slides for details)
            a_11 = (objCenter_PerturbX[0] - initObjCenter[0]) / 0.01
            a_21 = (objCenter_PerturbX[1] - initObjCenter[1]) / 0.01
            a_12 = (objCenter_PerturbY[0] - initObjCenter[0]) / 0.01
            a_22 = (objCenter_PerturbY[1] - initObjCenter[1]) / 0.01

            imageJacobian = np.array([[a_11, a_12], [a_21, a_22]])
            jPrime = np.linalg.inv(imageJacobian)
            J_PRIME = jPrime
        iteration += 1
        move_robot([adjustArmXMeters, adjustArmYMeters, 0, 0, 0, 0])
        time.sleep(1.5)
            

print("Calibrated Image Jacobian Matrix: ", imageJacobian)
print("Calibrated Inverse Image Jacobian Matrix: ", jPrime)

xPrev = 0
yPrev = 0  

iteration = 1

print()
print("STARTING VISUAL SERVOING ADJUSTMENTS")
print("************************************")
print("************************************")
print("************************************")
print("************************************")
print()

adjustArmXY = True

while adjustArmXY is True:
    print("Adjustment Iteration: ", iteration)

    boxes = get_berry_boxes()
    
    if boxes is None:
        print("Could not detect Blackberry")
        adjustArmXY = False
        break
    
    centroids = get_centroids_from_boxes(boxes)
    curr_centroid = centroids[0,:]

    print("CentroidX: ",curr_centroid[0])
    print("CentroidY: ",curr_centroid[1])

    imError = (curr_centroid - IMAGE_CENTER).reshape(-1,1)
    print("Image error vector: ")
    print(imError)

    deltaChange = 1/3* (jPrime @ imError)
    print("Delta change: ", deltaChange)

    newAdjustments = (-deltaChange)
    print("New Adjustments: ")
    print(newAdjustments)

    if newAdjustments[0][0] < 1e-4 and newAdjustments[1][0] < 1e-4:
        adjustArmXY = False
        print("XY adjusting finished, you are now aligned with object. Exiting while loop.")
    last_pos = trajectory[:,-1].reshape(-1,1)
    x_change_mm = 0
    y_change_mm = newAdjustments[0][0]*1000
    z_change_mm = newAdjustments[1][0]*1000
    change_mm = np.array([x_change_mm, y_change_mm, z_change_mm]).reshape(-1,1)
    curr_pos = last_pos + change_mm
    trajectory = np.concatenate((trajectory,curr_pos), axis=1)
    move_robot([newAdjustments[0][0], newAdjustments[1][0], 0, 0, 0, 0])
    time.sleep(1.5)

for i in range(3):
    boxes = get_berry_boxes()
    box_center = get_centroids_from_boxes(boxes)[0,:]
    width = boxes[0,2] - boxes[0,0]
    height = boxes[0,3] - boxes[0,1]
    length = (width + height) / 2
    corners = np.zeros((4,2))
    corners[0,:] = np.array([box_center[0] - length/2,box_center[1] - length/2])
    corners[1,:] = np.array([box_center[0] + length/2,box_center[1] - length/2])
    corners[2,:] = np.array([box_center[0] + length/2,box_center[1] + length/2])
    corners[3,:] = np.array([box_center[0] - length/2,box_center[1] + length/2])
    temp = np.zeros((1,4,2))
    temp[0] = corners
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(temp,17/1000,cameraMatrix,dist)
    depth_estimate = tvecs[0,0,2]
    depth_estimate *= 0.5
    print(depth_estimate)

    last_pos = trajectory[:,-1].reshape(-1,1)
    x_change_mm = depth_estimate * 1000
    y_change_mm = 0
    z_change_mm = 0
    change_mm = np.array([x_change_mm, y_change_mm, z_change_mm]).reshape(-1,1)
    curr_pos = last_pos + change_mm
    trajectory = np.concatenate((trajectory,curr_pos), axis=1)

    move_robot([0,0,depth_estimate,0,0,0])
    iteration += 1
    time.sleep(1.5)

move_robot([0,0,15/1000,0,0,0])

# cam = cv2.VideoCapture(0)
# cam.set(3, 1280)
# cam.set(4, 720)
# img = None
# for i in range(5):
#     result, img = cam.read()
# cam.release()
# cv2.imwrite("blackberry_img_buffer.jpg",img)
# boxes = my_detection("blackberry_img_buffer.jpg", YOLO_WEIGHT_FILE)
# box_center = get_centroids_from_boxes(boxes)[0,:]
# width = boxes[0,2] - boxes[0,0]
# height = boxes[0,3] - boxes[0,1]
# length = (width + height) / 2
# corners = np.zeros((4,2))
# corners[0,:] = np.array([box_center[0] - length/2,box_center[1] - length/2])
# corners[1,:] = np.array([box_center[0] + length/2,box_center[1] - length/2])
# corners[2,:] = np.array([box_center[0] + length/2,box_center[1] + length/2])
# corners[3,:] = np.array([box_center[0] - length/2,box_center[1] + length/2])
# temp = np.zeros((1,4,2))
# temp[0] = corners
# rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(temp,17/1000,cameraMatrix,dist)
# depth_estimate = tvecs[0,0,2]
# depth_estimate *= 0.5

# last_pos = trajectory[:,-1].reshape(-1,1)
# x_change_mm = depth_estimate * 1000
# y_change_mm = 0
# z_change_mm = 0
# change_mm = np.array([x_change_mm, y_change_mm, z_change_mm]).reshape(-1,1)
# curr_pos = last_pos + change_mm
# trajectory = np.concatenate((trajectory,curr_pos), axis=1)

# print(depth_estimate)
# move_robot([0,0,depth_estimate,0,0,0])
# time.sleep(1.5)


print("Adjustments are done.")
print("Ending program.")

print("\n")
print(trajectory)
print(trajectory.shape)
np.savetxt("trajectory.txt", trajectory, fmt='%.7f')