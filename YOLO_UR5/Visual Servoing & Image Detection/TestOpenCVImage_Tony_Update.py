import cv2
import copy
import numpy as np

def detect_circle(frame):
    img = copy.deepcopy(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3,3 ))
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred, 
                    cv2.HOUGH_GRADIENT_ALT, 1, 40, param1 = 300,
                param2 = 0.9, minRadius = 0, maxRadius = 150)
    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        detected_circles = detected_circles[0]
        for pt in detected_circles[0:5,:]:
            a, b, r = pt[0], pt[1], pt[2]
            t = (a,b)
            # Draw bound circle for the circle.
            cv2.rectangle(img, (a-r, b-r), (a+r, b+r), (0, 255, 0), 3)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 2, (0, 0, 255), -1)
            # Print centroid of the circle
            cv2.putText(img, str(t), (a, b-r), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,0),2, cv2.LINE_AA)
            
    return img


def main():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set width as 640
    cam.set(4, 480) # set height as 480
    count = 0
    while True:
        result, image = cam.read()
        count +=1
        if count >= 10 and result:
            output = detect_circle(image)
            cv2.imshow("Test", output)
            cv2.waitKey(5)
            cv2.destroyWindow("Test Image from USB Webcam")

main()