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

while adjustArmXY:

    detected_circles = None
    
    while detected_circles is None:

        # Image processing here
        # 0 if only one camera
        cam = cv2.VideoCapture(1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            result, image = cam.read()
            if result:
                # print("Image detected!")
                # saving image in local storage
                cv2.imwrite(("Image after %2d iterations.png" % (iteration)), image)
                break
            else: 
                print("Image Not Detected!")

        cam.release()

        # Read image
        img = cv2.imread(("Image after %2d iterations.png" % (iteration)), cv2.IMREAD_COLOR)

        output = img.copy()

        # Convert to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur using 3 * 3 kernel.
        gray_blurred = cv2.blur(gray, (3, 3))

        # Apply Hough transform on the blurred image.
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 50, param1 = 50, param2 = 30, minRadius = 10, maxRadius = 500)

        # Draw circles that are detected.
        if detected_circles is not None:
            # print("Detected %2d circle(s)." % (len(detected_circles)))

            # convert the (x, y) coordinates and radius of the circles to integers
            detected_circles = np.round(detected_circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in detected_circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 1)
                cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
                t = (x, y)
                cv2.putText(output, str(t), (x, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            print("Circle not detected. Restarting while loop to recapture image.")


    cv2.imwrite(("Detected Circles after %2d iterations.jpg" % (iteration)), output)
    # cv2.imshow(("Detected Circles %2d.jpg" % (iteration)), output)
    # cv2.waitKey()

    # Take first detected circle if more than one is detected (all pixel coordinates)
    centerX = detected_circles[0][0]
    centerY = detected_circles[0][1]
    radius = detected_circles[0][2]
    
    # print("Circle centroid x coordinate: ", centerX)
    # print("Circle centroid y coordinate: ", centerY)
    # print("Circle centroid radius: ", radius)

    circleArea = radius**2 * pi
    # print("Circle area in pixels squared: ", circleArea)

    # Default resolution for endoscope is 1280 x 720
    imageCenterX = 640
    imageCenterY = 360

    xError = -1 * (centerX - imageCenterX)
    print("Pixel error in the x direction: ", xError)

    # pixel frame y-axis is opposite of UR5 wrist frame IF setting wrist to have pos-y up, pos-x right, pos-z out of page
    yError = -1 * (centerY - imageCenterY)
    print("Pixel error in the y direction: ", yError)

    if abs(xError) > 3 or abs(yError) > 3:
        adjustArmXY = True
        print("Need to adjust arm.")
    else:
        adjustArmXY = False
        print("No need to adjust arm")

    # Convert pixel distances to real world distances for UR5 (hard coded conversion right now)
        # - Fixed distance of 8.125 inches from paper to UR5 wrist surface
        # - 59 pixels radius, 0.022225 meters radius IRL
        
    conversionFactor = 0.022225 / 59
    xErrorMeters = xError * conversionFactor
    yErrorMeters = yError * conversionFactor

    adjustArmXMeters =  (4 * xErrorMeters) / 5
    print("Arm is adjusting by", adjustArmXMeters, "meters in the x direction.")
    adjustArmYMeters = (4 * yErrorMeters) / 5
    print("Arm is adjusting by", adjustArmYMeters, "meters in the y direction.")

    i = 0
    while i < 1:
        i += 1
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
                print("XY Adjustment Iteration: ", iteration)
                time.sleep(0.5)

                # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
                pose = [adjustArmXMeters, adjustArmYMeters, 0, 0, 0, 0]

                values = str(pose[0]) + ", " + str(pose[1]) + ", " + str(pose[2]) + \
                    ", " + str(pose[3]) + ", " + str(pose[4]) + ", " + str(pose[5])
                poseString = "(" + values + ")"
                # print("   Pose string data to send: " + poseString)
                c.send(poseString.encode())

        except socket.error as socketerror:
            print(adjustArmXY)

print("Moving towards berry now.")

adjustArmZ = True

iteration = 0

camDistanceFromSphere = 100

while adjustArmZ:

    detected_circles = None
    
    while detected_circles is None:

        cam = cv2.VideoCapture(1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            result, image = cam.read()
            if result:
                # print("Distance Image detected!")
                cv2.imwrite(("DISTANCE Image after %2d iterations.png" % (iteration)), image)
                break
            else: 
                print("Image Not Detected!")

        cam.release()
        img = cv2.imread(("DISTANCE Image after %2d iterations.png" % (iteration)), cv2.IMREAD_COLOR)
        output = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))
        if camDistanceFromSphere < 90:
            detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 250, param1 = 50, param2 = 30, minRadius = 10, maxRadius = 500)
        elif camDistanceFromSphere < 50:
            detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 350, param1 = 50, param2 = 30, minRadius = 10, maxRadius = 500)
        else:
            detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 50, param1 = 50, param2 = 30, minRadius = 10, maxRadius = 500)
        if detected_circles is not None:
            # print("Detected %2d circle(s)." % (len(detected_circles)))
            detected_circles = np.round(detected_circles[0, :]).astype("int")
            for (x, y, r) in detected_circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 1)
                cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
                t = (x, y)
                cv2.putText(output, str(t), (x, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            print("Circle not detected. Restarting while loop to recapture image.")


    cv2.imwrite(("DISTANCE Detected Circles after %2d iterations.jpg" % (iteration)), output)
    centerX = detected_circles[0][0]
    centerY = detected_circles[0][1]
    radius = detected_circles[0][2]
    
    # print("Circle centroid x coordinate: ", centerX)
    # print("Circle centroid y coordinate: ", centerY)
    # print("Circle centroid radius: ", radius)

    circleArea = radius**2 * pi
    # print("Circle area in pixels squared: ", circleArea)

    # Find linear relationship between object radius in image and distance from object through calibration
        # For a radius of x = 54 pixels in the image, the distance was y = 204.7875 mm + (38.1mm because we had to move endoscope out about 1.5 inches to avoid fingers) - 83.2 mm - 38.1 = 121.5875 mm (from camera to flat edge of sphere).
        # Wrist distance is 204.7875 + 38.1, but distance from camera to sphere is about the same
        # For a radius of x = 88 pixels in the image, the distance was y = 71.5875 mm (from camera to flat edge of sphere).
        # Slope = (y2 - y1) / (x2 - x1) = -1.4706 mm/pixel, y-intercept = 201 mm
        # Linear Equation: y = -1.4706x + 201 mm

    camDistanceFromSphere = (-1.4706 * radius) + 201 # mm
    print("Distance from camera to sphere is: ", camDistanceFromSphere, " mm")
    distanceFromFingers = camDistanceFromSphere # mm
    print("Distance from fingers to sphere is: ", distanceFromFingers, " mm")
    
    if distanceFromFingers > 5:
        adjustArmZMeters = (distanceFromFingers / 2) / 1000
        print("Arm is adjusting by", adjustArmZMeters, "meters in the z direction.")
    else:
        adjustArmZ = False
        # adjustArmZMeters = 0.005 #adjust 5 more mm to to push fingers around sphere
        # print("Arm is adjusting by", adjustArmZMeters, "meters in the z direction for the last time.")

    i = 0
    while i < 1:
        i += 1
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
                print("Z Adjustment Iteration: ", iteration)
                time.sleep(0.5)

                # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
                pose = [0, 0, adjustArmZMeters, 0, 0, 0]

                values = str(pose[0]) + ", " + str(pose[1]) + ", " + str(pose[2]) + \
                    ", " + str(pose[3]) + ", " + str(pose[4]) + ", " + str(pose[5])
                poseString = "(" + values + ")"
                # print("   Pose string data to send: " + poseString)
                c.send(poseString.encode())

        except socket.error as socketerror:
            print(adjustArmZMeters)
c.close()
s.close()
print("Ending program.")
