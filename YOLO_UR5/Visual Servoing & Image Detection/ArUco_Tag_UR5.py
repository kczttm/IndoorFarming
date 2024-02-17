import numpy as np
from cv2 import aruco
import cv2
import socket

def build_transformation_matrix(rvec0, tvec0):
  R_0, _= cv2.Rodrigues(rvec0)
  T_0 = np.zeros((4,4))
  T_0[0:3,0:3] = R_0
  T_0[0:3,3] = tvec0[:]
  T_0[3,3] = 1
  return T_0


def get_tag_translation(aruco_size_in_mm):
  cameraMatrix = np.array([[927.31957258, 0,667.19142084],[0,922.20248778,335.69393703],[0,0,1]])
  dist = np.array([[-0.17574952,0.65288341, -0.00300312,  0.00724758, -0.95447869]])

  my_aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
  param =  aruco.DetectorParameters_create()
  param.cornerRefinementMethod = 1
  
  cam = cv2.VideoCapture(0)
  cam.set(3, 1280)
  cam.set(4, 720)
  centroid_aruco = np.zeros((2,))
  while True:
    try:
      ret, frame = cam.read()
      if ret:
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, my_aruco_dictionary,parameters=param)
          if ids is not None:
            # ArUco Centroid in Pixel Frame
            centroid_aruco = np.average(corners[0][0,:,:], axis=0)
            
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, aruco_size_in_mm/1000, cameraMatrix, dist)
            R0 = rvecs[0]
            t0 = tvecs[0]
            T0 = build_transformation_matrix(R0, t0)

            # Print Depth here
            print("Depth: ",t0[0,2])
          cv2.circle(frame, tuple(np.int64(centroid_aruco)), 10, (0,0,255), -1)
          cv2.imshow("",frame)
          cv2.waitKey(10)
    except KeyboardInterrupt:
      break

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
  return centroid_aruco, depth

frame = cv2.imread("Test_img1.jpg")
centroid_aruco, depth = get_centriod_and_depth_from_frame(frame, 10)
print("Centroid of ArUco: ", centroid_aruco)
print("Depth: ", depth)