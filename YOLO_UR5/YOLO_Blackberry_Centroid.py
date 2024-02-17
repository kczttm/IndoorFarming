import time
from yolov5.detect import yolo_detection
import numpy as np
import cv2

YOLO_WEIGHT_FILE = "/Users/tonytu/Desktop/Soft_Robotics_Internship/YOLO_UR5_Test/yolov5/runs/train/exp11/weights/best.pt"
cameraMatrix = np.array([[927.31957258, 0,667.19142084],[0,922.20248778,335.69393703],[0,0,1]])
dist = np.array([[-0.17574952,0.65288341, -0.00300312,  0.00724758, -0.95447869]])

def get_centroids_from_boxes(boxes):
    centroids = np.zeros((boxes.shape[0], 2))
    centroids[:,0] = (boxes[:,0] + boxes[:,2]) / 2
    centroids[:,1] = (boxes[:,1] + boxes[:,3]) / 2
    return np.int64(centroids)
def get_leftmost_centroid(centroids):
    sorted_ccentroids = centroids[centroids[:, 0].argsort()]
    return sorted_ccentroids[0,:]

boxes = yolo_detection("/Users/tonytu/Desktop/Soft_Robotics_Internship/YOLO_UR5_Test/Level_B_Images/B_img_4.jpg", YOLO_WEIGHT_FILE)
img = cv2.imread("/Users/tonytu/Desktop/Soft_Robotics_Internship/YOLO_UR5_Test/Level_B_Images/B_img_4.jpg")
box_center = get_centroids_from_boxes(boxes)
print(box_center)
print(box_center[:, 0].argsort())


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
# cv2.rectangle(img,tuple(np.int64(corners[0,:])),tuple(np.int64(corners[2,:])),(0,255,0),5)
# cv2.putText(img=img, text=str(np.round(depth_estimate,4)) + "m", org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=3)
# cv2.imwrite("blackberry_img_buffer.jpg",img)
# print(depth_estimate)