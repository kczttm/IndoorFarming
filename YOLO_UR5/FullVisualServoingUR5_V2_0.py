import socket
import time
import cv2
from cv2 import aruco
import numpy as np
from cmath import pi
import time

def build_transformation_matrix(rvec0, tvec0):
  R_0, _= cv2.Rodrigues(rvec0)
  T_0 = np.zeros((4,4))
  T_0[0:3,0:3] = R_0
  T_0[0:3,3] = tvec0[:]
  T_0[3,3] = 1
  return T_0


def get_centriod_and_depth_from_frame(frame, aruco_size_in_mm):
    cameraMatrix = np.array([[927.31957258, 0,667.19142084],[0,922.20248778,335.69393703],[0,0,1]])
    dist = np.array([[-0.17574952,0.65288341, -0.00300312,  0.00724758, -0.95447869]])

    my_aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    param =  aruco.DetectorParameters_create()
    param.cornerRefinementMethod = 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, my_aruco_dictionary,parameters=param)
    if ids is not None:
        # ArUco Centroid in Pixel Frame, numpy.ndarray of shape (2,)
        centroid_aruco = np.average(corners[0][0,:,:], axis=0)
        
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, aruco_size_in_mm/1000, cameraMatrix, dist)
        R0 = rvecs[0]
        t0 = tvecs[0]
        T0 = build_transformation_matrix(R0, t0)

        # Print Depth here (Scalar)
        depth = t0[0,2]
        return True, centroid_aruco, depth
    else:
        return False, np.array([-1.0,-1.0]), -1


HOST = "169.254.14.134"  # Laptop (server) IP address.
PORT = 30002

print("Starting Python program:")

adjustArm = True
iteration = 0

# Calibration to Find Image Jacobian Matrix
while iteration < 5:

    print("Calibration Iteration: ", iteration)
    # centroid_aruco = np.zeros((2,))
    # depth = -1.0
    
    cam = cv2.VideoCapture(1)
    cam.set(3, 1280)
    cam.set(4, 720)

    counter = 0

    while True:
        result, image = cam.read()
        print(result)
        if counter == 10:
            if result:
                cv2.imwrite(("Calibration %2d.png" % (iteration)), image)
                break
            else: 
                 print("Image Not Detected!")
        counter += 1

    cam.release()

    img = cv2.imread(("Calibration %2d.png" % (iteration)), cv2.IMREAD_COLOR)

    is_Detected_Aruco, centroid_aruco, depth = get_centriod_and_depth_from_frame(img, 10)
    # cv2.circle(img, tuple(np.int64(centroid_aruco)), 10, (0,0,255), -1)
    # cv2.putText(img,str(tuple(np.int64(centroid_aruco))),(200, 200),cv2.FONT_HERSHEY_DUPLEX,3.0,(125, 246, 55),3)

    # cv2.imwrite(("Centroid Calibration %2d.png" % (iteration)), img)

    if is_Detected_Aruco is False:
        print("Could not detect Aruco Tag")
        print("Trying again")
    else:
        tagCentroidX = centroid_aruco[0]
        tagCentroidY = centroid_aruco[1]
        depthFromEndoscope = depth

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

        iteration += 1

        i = 0
        while i < 1:
            i += 1
            # Communicate with UR5
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))  # Bind to the port.
            s.listen(5)  # Wait for UR5 (client) connection.
            c, addr = s.accept()  # Establish connection with client.

            try:
                msg = c.recv(1024).decode()  # Receive message from UR5.
                if msg == 'UR3_is_asking_for_data':
                    print("   UR5 is asking for data...")
                    time.sleep(0.5)
                    print("Moving ", adjustArmXMeters, " meters in the x direction and ", adjustArmYMeters, " meters in the y direction.")
                    
                    # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
                    pose = [adjustArmXMeters, adjustArmYMeters, 0, 0, 0, 0]
                    values = str(pose[0]) + ", " + str(pose[1]) + ", " + str(pose[2]) + \
                        ", " + str(pose[3]) + ", " + str(pose[4]) + ", " + str(pose[5])
                    poseString = "(" + values + ")"
                    print(poseString)
                    c.send(poseString.encode())

            except socket.error as socketerror:
                print("Error occured.")

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
adjustArmZ = True

while adjustArmXY is True or adjustArmZ is True:

    while adjustArmXY:

        print("Adjustment Iteration: ", iteration)

        cam = cv2.VideoCapture(1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)       
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        time.sleep(3)

        counter = 0
        while True:
            result, image = cam.read()
            if counter == 10:
                if result:
                    cv2.imwrite(("Adjusting Image after %2d iterations.png" % (iteration)), image)
                    break
                else: 
                    print("Image Not Detected!")
            counter += 10

        cam.release()

        img = cv2.imread(("Adjusting Image after %2d iterations.png" % (iteration)), cv2.IMREAD_COLOR)
        is_Detected_Aruco, centroid_aruco, depth = get_centriod_and_depth_from_frame(img, 10)
        # cv2.circle(img, tuple(np.int64(centroid_aruco)), 10, (0,0,255), -1)
        # cv2.putText(img,str(tuple(np.int64(centroid_aruco))),(200, 200),cv2.FONT_HERSHEY_DUPLEX,3.0,(125, 246, 55),3)
        # cv2.imwrite(("Centroid Adjusting Image after %2d iterations.png" % (iteration)), img)
        
        if is_Detected_Aruco is False:
            print("Could not detect Aruco Tag")
            adjustArmXY = False
            break
        
        tagCentroidX = centroid_aruco[0]
        tagCentroidY = centroid_aruco[1]
        depthFromEndoscope = depth

        print("tagCentroidX: ",tagCentroidX)
        print("tagCentroidY: ",tagCentroidY)

        imError = np.array([[tagCentroidX - 640], [tagCentroidY - 360]])
        print("Image error vector: ")
        print(imError)

        deltaChange = 1/3 * np.array([[(jPrime[0][0] * imError[0][0]) + (jPrime[0][1] * imError[1][0])], [(jPrime[1][0] * imError[0][0]) + (jPrime[1][1] * imError[1][0])]])
        print("Delta change: ", deltaChange)

        newAdjustments = np.array([[0 - deltaChange[0][0]], [0 - deltaChange[1][0]]])
        print("New Adjustments: ")
        print(newAdjustments)

        if newAdjustments[0][0] < 1e-4 and newAdjustments[1][0] < 1e-4:
            adjustArmXY = False
            print("XY adjusting finished, you are now aligned with object. Exiting while loop.")

        i = 0
        while i < 1:
            i += 1
            # Communicate with UR5
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))  # Bind to the port.
            s.listen(5)  # Wait for UR5 (client) connection.
            c, addr = s.accept()  # Establish connection with client.

            try:
                msg = c.recv(1024).decode()  # Receive message from UR5.
                if msg == 'UR3_is_asking_for_data':

                    print("   UR5 is asking for data...")
                    iteration += 1
                    time.sleep(0.5)
                    print("Moving ", newAdjustments[0][0], " meters in the x direction and ", newAdjustments[1][0], " meters in the y direction.")

                    # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
                    pose = [newAdjustments[0][0], newAdjustments[1][0], 0, 0, 0, 0]

                    values = str(pose[0]) + ", " + str(pose[1]) + ", " + str(pose[2]) + \
                        ", " + str(pose[3]) + ", " + str(pose[4]) + ", " + str(pose[5])
                    poseString = "(" + values + ")"
                    print("   Pose string data to send: " + poseString)
                    c.send(poseString.encode())

            except socket.error as socketerror:
                print("Error Occured")
    
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)       
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    result, image = cam.read()
    is_Detected_Aruco, centroid_aruco, depth = get_centriod_and_depth_from_frame(image, 10)
    # Z Adjustment 
    if depth <= 0.025:
        adjustArmZ = False
        print("Arm is within 0.025 meters of tag in the Z direction.")
        
        # Move in a little more
        i = 0
        while i < 1:
            i += 1
            # Communicate with UR5
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))  # Bind to the port.
            s.listen(5)  # Wait for UR5 (client) connection.
            c, addr = s.accept()  # Establish connection with client.

            try:
                msg = c.recv(1024).decode()  # Receive message from UR5.
                if msg == 'UR3_is_asking_for_data':

                    print("   UR5 is asking for data...")
                    iteration += 1
                    time.sleep(2)
                    print("Z-axis movement: ", 0.02)

                    # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
                    pose = [0, 0, 0.02, 0, 0, 0]

                    values = str(pose[0]) + ", " + str(pose[1]) + ", " + str(pose[2]) + \
                        ", " + str(pose[3]) + ", " + str(pose[4]) + ", " + str(pose[5])
                    poseString = "(" + values + ")"
                    print("   Pose string data to send: " + poseString)
                    c.send(poseString.encode())

            except socket.error as socketerror:
                print("Error Occured")

    else:
        if depth < 0:
            continue
        zAdjustment = depth / 2
        print("Arm is adjusting ", zAdjustment, " meters")
        i = 0
        while i < 1:
            i += 1
            # Communicate with UR5
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))  # Bind to the port.
            s.listen(5)  # Wait for UR5 (client) connection.
            c, addr = s.accept()  # Establish connection with client.

            try:
                msg = c.recv(1024).decode()  # Receive message from UR5.
                if msg == 'UR3_is_asking_for_data':

                    print("   UR5 is asking for data...")
                    iteration += 1
                    time.sleep(0.5)
                    print("Z-axis movement: ", zAdjustment)
                    # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
                    pose = [0, 0, zAdjustment, 0, 0, 0]

                    values = str(pose[0]) + ", " + str(pose[1]) + ", " + str(pose[2]) + \
                        ", " + str(pose[3]) + ", " + str(pose[4]) + ", " + str(pose[5])
                    poseString = "(" + values + ")"
                    print("   Pose string data to send: " + poseString)
                    c.send(poseString.encode())

            except socket.error as socketerror:
                print("Error Occured")

print("Adjustments are done.")

# Gripping Command


c.close()
s.close()
print("Ending program.")
