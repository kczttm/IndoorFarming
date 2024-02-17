import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)
img = None
for i in range(10):
    result, img = cam.read()
    cv2.imwrite("4.jpg", img)
