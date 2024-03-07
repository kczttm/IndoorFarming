import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to draw optical flow
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_, _) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

# Function to draw HSV representation of optical flow
def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

# Load images
img1 = cv2.imread('image1.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('image2.png')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect good features to track
p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.02, minDistance=7, blockSize=7)

# Calculate optical flow using Lucas-Kanade method
p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None)

# Select good points
good_new = p1[st == 1]
good_old = p0[st == 1]

# Draw the tracks
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    a, b, c, d = int(a), int(b), int(c), int(d)  # Convert to integers
    img1 = cv2.circle(img1, (a, b), 5, (255, 0, 0), -1)
    img1 = cv2.line(img1, (a, b), (c, d), (0, 255, 0), 2)

# Display the images with the tracks
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.show()
