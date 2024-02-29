import socket
import time
import cv2
import numpy as np
from cmath import pi

HOST = "169.254.14.134"  # Laptop (server) IP address.
PORT = 30002

print("Starting Python program:")

adjustArmXY = True
iteration = 0

# Calibration to find image Jacobian Matrix
while iteration < 5:

    print("Calibration Iteration: ", iteration)
    detected_circles = None
    
    while detected_circles is None:
        cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        counter = 0

        while True:
            result, image = cam.read()
            if counter == 10:
                if result:
                    cv2.imwrite(("Calibration %2d.png" % (iteration)), image)
                    break
                else: 
                    print("Image Not Detected!")
            counter += 1

        cam.release()
        img = cv2.imread(("Calibration %2d.png" % (iteration)), cv2.IMREAD_COLOR)
        output = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 150, param1 = 50, param2 = 30, minRadius = 10, maxRadius = 500)

        if detected_circles is not None:
            detected_circles = np.round(detected_circles[0, :]).astype("int")
            for (x, y, r) in detected_circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 1)
                cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
                t = (x, y)
                cv2.putText(output, str(t), (x, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            print("Circle not detected. Restarting while loop to recapture image.")

    cv2.imwrite(("Calibration Detected Circles %2d.jpg" % (iteration)), output)
    cv2.destroyAllWindows()

    print("Detected %.2f circle(s)." %(len(detected_circles)))

    if iteration == 0:
        initObjCenter = [detected_circles[0][0], detected_circles[0][1]]
        print("P0: ", initObjCenter)
        perturbX = 0.01
        perturbY = 0
        adjustArmXMeters = perturbX
        adjustArmYMeters = perturbY

    elif iteration == 1:
        objCenter_PerturbX = [detected_circles[0][0], detected_circles[0][1]]
        print("P1: ", objCenter_PerturbX)
        adjustArmXMeters = perturbX * -1 # Return to initial pose
        adjustArmYMeters = 0

    elif iteration == 2:
        perturbX = 0
        perturbY = 0.01       
        adjustArmXMeters = perturbX
        adjustArmYMeters = perturbY
    
    elif iteration == 3:
        objCenter_PerturbY = [detected_circles[0][0], detected_circles[0][1]]
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

while adjustArmXY:

    print("XY Adjustment Iteration: ", iteration)

    detected_circles = None
    
    while detected_circles is None:

        cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
        output = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 150, param1 = 50, param2 = 30, minRadius = 10, maxRadius = 500)
        
        if detected_circles is not None:

            detected_circles = np.round(detected_circles[0, :]).astype("int")
            for (x, y, r) in detected_circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 1)
                cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
                t = (x, y)
                cv2.putText(output, str(t), (x, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            print("Circle not detected. Restarting while loop to recapture image.")

    print("Detected %.2f circle(s)." %(len(detected_circles)))
    cv2.imwrite(("Adjusting Detected Circles after %2d iterations.jpg" % (iteration)), output)
    cv2.destroyAllWindows()

    centerX = detected_circles[0][0]
    centerY = detected_circles[0][1]

    imError = np.array([[centerX - 640], [centerY - 360]])
    print("Image error vector: ")
    print(imError)

    deltaChange = np.array([[(jPrime[0][0] * imError[0][0]) + (jPrime[0][1] * imError[1][0])], [(jPrime[1][0] * imError[0][0]) + (jPrime[1][1] * imError[1][0])]])
    print("Delta change: ", deltaChange)

    newAdjustments = np.array([[0 - deltaChange[0][0]], [0 - deltaChange[1][0]]])
    print("New Adjustments: ")
    print(newAdjustments)

    if newAdjustments[0][0] == 0 and newAdjustments[1][0] == 0:
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
                # print("   Pose string data to send: " + poseString)
                c.send(poseString.encode())

        except socket.error as socketerror:
            print("Error Occured")

########################################################################################################
# Approaching berry in Z direction 

# print("Moving towards berry now.")

# adjustArmZ = True

# iteration = 0

# camDistanceFromSphere = 100

# while adjustArmZ:

#     detected_circles = None
    
#     while detected_circles is None:

#         cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
#         cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # counter = 0

        # while True:
        #     result, image = cam.read()
        #         if counter == 10:
        #             if result:
        #                 # print("Distance Image detected!")
        #                 cv2.imwrite(("DISTANCE Image after %2d iterations.png" % (iteration)), image)
        #                 break
        #             else: 
        #                 print("Image Not Detected!")
        #       counter += 1

#         cam.release()
#         img = cv2.imread(("DISTANCE Image after %2d iterations.png" % (iteration)), cv2.IMREAD_COLOR)
#         output = img.copy()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray_blurred = cv2.blur(gray, (3, 3))
#         if camDistanceFromSphere < 90:
#             detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 250, param1 = 50, param2 = 30, minRadius = 10, maxRadius = 500)
#         elif camDistanceFromSphere < 50:
#             detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 350, param1 = 50, param2 = 30, minRadius = 10, maxRadius = 500)
#         else:
#             detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 50, param1 = 50, param2 = 30, minRadius = 10, maxRadius = 500)
#         if detected_circles is not None:
#             # print("Detected %2d circle(s)." % (len(detected_circles)))
#             detected_circles = np.round(detected_circles[0, :]).astype("int")
#             for (x, y, r) in detected_circles:
#                 cv2.circle(output, (x, y), r, (0, 255, 0), 1)
#                 cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
#                 t = (x, y)
#                 cv2.putText(output, str(t), (x, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
#         else:
#             print("Circle not detected. Restarting while loop to recapture image.")


#     cv2.imwrite(("DISTANCE Detected Circles after %2d iterations.jpg" % (iteration)), output)
#     objectCenterX = detected_circles[0][0]
#     objectCenterY = detected_circles[0][1]
#     radius = detected_circles[0][2]
    
#     # print("Circle centroid x coordinate: ", centerX)
#     # print("Circle centroid y coordinate: ", centerY)
#     # print("Circle centroid radius: ", radius)

#     circleArea = radius**2 * pi
#     # print("Circle area in pixels squared: ", circleArea)

#     # Find linear relationship between object radius in image and distance from object through calibration
#         # For a radius of x = 54 pixels in the image, the distance was y = 204.7875 mm + (38.1mm because we had to move endoscope out about 1.5 inches to avoid fingers) - 83.2 mm - 38.1 = 121.5875 mm (from camera to flat edge of sphere).
#         # Wrist distance is 204.7875 + 38.1, but distance from camera to sphere is about the same
#         # For a radius of x = 88 pixels in the image, the distance was y = 71.5875 mm (from camera to flat edge of sphere).
#         # Slope = (y2 - y1) / (x2 - x1) = -1.4706 mm/pixel, y-intercept = 201 mm
#         # Linear Equation: y = -1.4706x + 201 mm

#     camDistanceFromSphere = (-1.4706 * radius) + 201 # mm
#     print("Distance from camera to sphere is: ", camDistanceFromSphere, " mm")
#     distanceFromFingers = camDistanceFromSphere # mm
#     print("Distance from fingers to sphere is: ", distanceFromFingers, " mm")
    
#     if distanceFromFingers > 5:
#         adjustArmZMeters = (distanceFromFingers / 2) / 1000
#         print("Arm is adjusting by", adjustArmZMeters, "meters in the z direction.")
#     else:
#         adjustArmZ = False
#         # adjustArmZMeters = 0.005 #adjust 5 more mm to to push fingers around sphere
#         # print("Arm is adjusting by", adjustArmZMeters, "meters in the z direction for the last time.")

#     i = 0
#     while i < 1:
#         i += 1
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         s.bind((HOST, PORT))  # Bind to the port.
#         s.listen(5)  # Wait for UR5 (client) connection.
#         c, addr = s.accept()  # Establish connection with client.

#         try:
#             msg = c.recv(1024).decode()  # Receive message from UR5.
#             if msg == 'UR3_is_asking_for_data':

#                 print("   UR5 is asking for data...")
#                 iteration += 1
#                 print("Z Adjustment Iteration: ", iteration)
#                 time.sleep(0.5)

#                 # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
#                 pose = [0, 0, adjustArmZMeters, 0, 0, 0]

#                 values = str(pose[0]) + ", " + str(pose[1]) + ", " + str(pose[2]) + \
#                     ", " + str(pose[3]) + ", " + str(pose[4]) + ", " + str(pose[5])
#                 poseString = "(" + values + ")"
#                 # print("   Pose string data to send: " + poseString)
#                 c.send(poseString.encode())

#         except socket.error as socketerror:
#             print(adjustArmZMeters)

c.close()
s.close()
print("Ending program.")
