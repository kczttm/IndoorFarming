from cmath import pi
import cv2
import numpy as np

# 0 if only one camera
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

counter = 0

while True: 
    result, image = cam.read()
    if counter == 10:
        if result:
            # saving image in local storage
            cv2.imwrite("Captured Image.png", image)
            break
        else:
            print("no image captured")

    counter += 1

# Read image
img = cv2.imread("Captured Image.png", cv2.IMREAD_COLOR)

output = img.copy()

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3, 3))

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 150, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 300)

# Draw circles that are detected.
if detected_circles is not None:
    print("Detected a circle!")

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
    print("No circle detected.")

cv2.imwrite("Detected Circles.jpg", output)
cv2.imshow("Detected Circles", output)
cv2.waitKey()


# # Take first detected circle if more than one is detected (all pixel coordinates)
# centerX = detected_circles[0][0]
# centerY = detected_circles[0][1]
# radius = detected_circles[0][2]

# print("Circle centroid x coordinate: ", centerX)
# print("Circle centroid y coordinate: ", centerY)
# print("Circle centroid radius: ", radius)

# circleArea = radius**2 * pi
# print("Circle area in pixels squared: ", circleArea)

# # Default resolution for endoscope is 640 x 480p 
# imageCenterX = 320
# imageCenterY = 240

# xError = centerX - imageCenterX
# print("Pixel error in the x direction: ", xError)

# yError = -1 * (centerY - imageCenterY) # pixel frame y-axis is opposite of UR5 wrist frame IF setting wrist to have pos-y up, pos-x right, pos-z out of page
# print("Pixel error in the y direction: ", yError)

# if abs(xError) > 1 or abs(yError) > 1:
#     adjustArm = True
#     print("Need to adjust arm.")
# else: 
#     adjustArm = False
#     print("No need to adjust arm")

# # Convert pixel distances to real world distances for UR5 (hard coded conversion right now)
# conversionFactor = 1
# xErrorMeters = xError * conversionFactor
# yErrorMeters = yError * conversionFactor

# armAdjustXMeters = xErrorMeters / 3
# armAdjustYMeters = yErrorMeters / 3