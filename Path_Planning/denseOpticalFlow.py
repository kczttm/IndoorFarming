import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_flow(img, flow, step=16):

    height, width = img.shape[:2]
    y, x = np.mgrid[step/2:height:step, step/2:width:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))
    
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr

def draw_hsv(flow):

    height, width = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx + fy*fy)

    hsv = np.zeros((height, width, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 10, 255) # Scale v higher to increase contrast in colors
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

image1 = cv2.imread('image5.png')
gray_scaled_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

image2 = cv2.imread('image6.png')
gray_scaled_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Resize the second image to match the width of the first image
height, width = gray_scaled_image1.shape
gray_scaled_image2 = cv2.resize(gray_scaled_image2, (width, height))

flow = cv2.calcOpticalFlowFarneback(gray_scaled_image1, gray_scaled_image2, None, 0.1, 10, 50, 7, 7, 1.2, 0)
cv2.imshow('Flow', draw_flow(gray_scaled_image2, flow))
cv2.imshow('Flow HSV', draw_hsv(flow))

# Wait for a key press and close the windows when a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
