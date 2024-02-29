import cv2

cam = cv2.VideoCapture(1)
cam.set(3, 1280)
cam.set(4, 720)
result, img = cam.read()
cv2.imwrite("2_Berry_Leaf_2.5offset.png", img)