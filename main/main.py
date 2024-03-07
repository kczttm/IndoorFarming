import open3d as o3d
import os
import time
import sys
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
import plotly.express as px
import plotly.subplots as subplots
import plotly.graph_objects as go

sys.path.append('../RAFT/core') # Add to path
sys.path.append('../YOLO_UR5') # Add to path
sys.path.append('../YOLO_UR5/utils')

# Add the parent directory of 'main' to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname('main.ipynb'), os.pardir))
sys.path.append(os.path.join(parent_dir, 'RAFT'))

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
from helper_functions import *

print("Is torch.cuda available? ", torch.cuda.is_available())
print("Device Count: ",torch.cuda.device_count())

if torch.cuda.is_available():
    curr_device = 'cuda'
    print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    curr_device = 'cpu'

# curr_dir = os.path.dirname(os.path.realpath(__file__))
# yolo_directory = os.path.join(curr_dir, 'YOLO_UR5')
# sys.path.add(yolo_directory)

# import run_YOLO

# %matplotlib inline
# annotated_image_1, berry_centroids_1 = YOLO.apply_yolo_to_image('2.5mm_1.png')
# annotated_image_2, berry_centroids_2 = YOLO.apply_yolo_to_image('2.5mm_2.png')

# # print(berry_centroids_1)
# # print(berry_centroids_2)

# # print(annotated_image_1.shape)
# # print(annotated_image_2.shape)

# image_center_x = annotated_image_1.shape[1] / 2
# image_center_y = annotated_image_1.shape[0] / 2

# # print(image_center_x)
# # print(image_center_y)

# frame_1_centroid_x = berry_centroids_1[:, 0]
# frame_1_centroid_y = berry_centroids_1[:, 1]

# # print(frame_1_centroid_x)
# # print(frame_1_centroid_y)

# x_disparity = np.abs(berry_centroids_2 - berry_centroids_1)[:, 0]

# # print(x_disparity)

# # berry1_x = frame_1_centroid_x[0]
# # berry1_y = frame_1_centroid_y[0]

# berries_x_pos, berries_y_pos, berries_z_pos = pix_to_pos(frame_1_centroid_x, frame_1_centroid_y, x_disparity, image_center_x, image_center_y)
# berries_pos_matrix = np.column_stack((berries_x_pos, berries_y_pos, berries_z_pos))
# print("Berry Positions (m): ")
# print(berries_pos_matrix)

# fig, axes = plt.subplots(1, 2, figsize=(15, 10))
# axes[0].imshow(annotated_image_1)
# axes[0].set_title('YOLO Berry Detection - Image 1')
# axes[1].imshow(annotated_image_2)
# axes[1].set_title('YOLO Berry Detection - Image 2')
# plt.tight_layout()
# plt.show()
print("what")
model = load_model("../RAFT/models/raft-sintel.pth", args=Args())
frame1 = cv2.imread('Frame1_5mm.png')
frame2 = cv2.imread('Frame2_5mm.png')

print("hi")

# Downsample image to reduce computation complexity (using Lanczos4 downsampling algorithm)
frame1 = cv2.resize(frame1, (320, 240), interpolation=cv2.INTER_LANCZOS4)
frame2 = cv2.resize(frame2, (320, 240), interpolation=cv2.INTER_LANCZOS4)

print(frame1.shape)

# Convert images to rgb color space
frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

# Resize the second image to match the dimensions of the first image
height, width, channels = frame2.shape
frame1 = cv2.resize(frame1_rgb, (width, height))

# Apply RAFT Optical Flow Model on Frame 1 and 2
time1 = time.perf_counter()
flow_iters = inference(model, frame1, frame2_rgb, device=curr_device, iters=5, test_mode=False)
time2 = time.perf_counter()
print("Inference Time: ", time2 - time1)

fig2, axes = plt.subplots(3, 2, figsize=(18, 12), gridspec_kw={'height_ratios': [4, 4, 1]})
fig2.subplots_adjust(hspace=0.3)
axes[0, 0].imshow(frame1)
axes[0, 0].set_title("Frame 1")
axes[0, 1].imshow(frame2)
axes[0, 1].set_title("Frame 2")

first_flow_viz = axes[1, 0].imshow(get_viz(flow_iters[0]))
axes[1, 0].set_title('First RAFT Flow Iteration (Frame 1)')
axes[1, 0].set_xlabel("Pixels"); axes[1, 0].set_ylabel("Pixels")
final_flow_viz = axes[1, 1].imshow(get_viz(flow_iters[-1]))
axes[1, 1].set_title('Final RAFT Flow Results (Frame 1)')
axes[1, 1].set_xlabel("Pixels"); axes[1, 1].set_ylabel("Pixels")

# Add color bars
axes[2, 0].axis('off'); axes[2, 1].axis('off')
cbar0 = fig2.colorbar(first_flow_viz, ax=axes[2, :], fraction=0.4, pad=0.04, location="top")
cbar0.set_label('Magnitude of Optical Flow')
plt.show()

# Retrieve final flow iteration and the corresponding flow tensors for x and y
final_flow = flow_iters[-1]
flow_x_matrix = final_flow[0, 0]  # Horizontal displacement component in pixels
flow_y_matrix = final_flow[0, 1]  # Vertical displacement component in pixels

# Compute the magnitude of flow
flow_magnitude = torch.sqrt(flow_x_matrix**2 + flow_y_matrix**2)

# Convert flow tensors to numpy arrays and round all points to 4 decimal places
flow_x_matrix = np.round(flow_x_matrix.cpu().numpy(), decimals=4) # Tensor to numpy array can only be done on CPU
flow_y_matrix = np.round(flow_y_matrix.cpu().numpy(), decimals=4)
flow_magnitude = np.round(flow_magnitude.cpu().numpy(), decimals=4)

np.savetxt('flow_x_matrix.txt', flow_x_matrix)
np.savetxt('flow_y_matrix.txt', flow_y_matrix)
np.savetxt('flow_magnitude.txt', flow_magnitude)

# Extra Information
max_index = np.unravel_index(np.argmax(flow_x_matrix, axis=None), flow_x_matrix.shape)
print("Index of maximum flow-x value:", max_index)
baseline_x = 0.005
tilt = np.radians(0) 
print("Maximum Flow in x-direction :", np.max(flow_x_matrix), "pixels")
print("Minimum Flow in x-direction :", np.min(flow_x_matrix), "pixels")

# Calculating pixel to world conversion
world_conversion = baseline_x * np.cos(tilt) / np.max(flow_x_matrix) 
print("Pixels per mm at the depth where max disparity is: ", (1 / world_conversion) / 1000)

# Apply threshold to flow magnitude
flow_threshold = 12  # Pixel displacement threshold value
flow_magnitude_filtered = np.where(flow_magnitude < flow_threshold, 0, flow_magnitude) # New matrix of pixels that turn black (value set to 0) if not passing flow threshold
removed_indices = np.where(flow_magnitude_filtered == 0)

# Visualize the thresholded flow magnitude
plt.imshow(flow_magnitude_filtered, cmap='gray')
plt.title('Thresholded Flow Magnitude')
plt.colorbar()
plt.show()

# Adjust Flow matrices so that the filtered out pixels are set to infinity so they can be ignored for later processing
flow_x_matrix[removed_indices] = np.inf
flow_y_matrix[removed_indices] = np.inf

# Creating Matrix of Pixels
height, width, channels = final_flow_viz.get_array().shape

# Create arrays representing the x and y coordinates of pixels
x_coords = np.tile(np.arange(width), height)
y_coords = np.repeat(np.arange(height), width)

# Reshape x_coords and y_coords to match the shape of the image
x_coords = x_coords.reshape(height, width)
y_coords = y_coords.reshape(height, width)

# Stack x_coords and y_coords to create the matrix of pixels
pixel_matrix = np.dstack((x_coords, y_coords))
x_pix_matrix = pixel_matrix[:, :, 0]
y_pix_matrix = pixel_matrix[:, :, 1]

# print("X Pixel Matrix Shape: ", x_pix_matrix.shape)
# print("Y Pixel Matrix Shape: ", y_pix_matrix.shape)

# Get matrices of (x, y, z) locations for each pixel in image
im_center_x = final_flow[0][0].shape[1] / 2
print(im_center_x)
im_center_y = final_flow[0][0].shape[0] / 2 
print(im_center_y)
x_pos_matrix, y_pos_matrix, z_pos_matrix = pix_to_pos(x_pix_matrix, y_pix_matrix, flow_x_matrix, im_center_x, im_center_y) 
print(z_pos_matrix.shape)
kept_indices = np.where(np.isfinite(z_pos_matrix))

x_pos_filtered = x_pos_matrix[kept_indices]
y_pos_filtered = y_pos_matrix[kept_indices]
z_pos_filtered = z_pos_matrix[kept_indices]     
obstacle_3d_points = np.column_stack((x_pos_filtered, y_pos_filtered, z_pos_filtered))

print("Obstacle Points Shape: ", obstacle_3d_points.shape)

print("\nMaximum finite x value (m): ", np.round(np.max(x_pos_matrix[np.isfinite(x_pos_matrix)]), decimals=4))
print("Minimum x value (m): ", np.round(np.min(x_pos_matrix), decimals=4))

print("\nMaximum finite y value (m): ", np.round(np.max(y_pos_matrix[np.isfinite(y_pos_matrix)]), decimals=4))
print("Minimum y value (m): ", np.round(np.min(y_pos_matrix), decimals=4))

print("\nMaximum finite z value (m): ", np.round(np.max(z_pos_matrix[np.isfinite(z_pos_matrix)]), decimals=4))
print("Minimum depth value (m): ", np.round(np.min(z_pos_matrix), decimals=4))

# Create 3D point cloud plot using plotly and 2D scatter plot of XY positions
fig_3d = px.scatter_3d(x=x_pos_filtered, y=y_pos_filtered, z=z_pos_filtered, color=z_pos_filtered,
                            opacity=1, size_max=5, color_continuous_scale='Viridis')
fig_3d.update_layout(title='3D Point Cloud', coloraxis_colorbar=dict(title='Z_Depth Magnitude'), width=1000, height=500)
fig_3d.show()

fig_2d = px.scatter(x=x_pos_filtered, y=y_pos_filtered)
fig_2d.update_layout(title='XY View of Point Cloud')
fig_2d.show()

# Create Octree Representation of Point Cloud Information using Open3d
# pcd = o3d.geometry.PointCloud()
# points = o3d.utility.Vector3dVector(obstacle_3d_points) # Convert numpy array to Open3D Vector
# pcd.points = points

# frame1_colors = frame1[kept_indices] # Extracting pixels that are kept after threshold 
# frame1_colors_norm = frame1_colors / 255.0 # Normalize color values to [0, 1]
# print(frame1_colors_norm[0])

# colors = o3d.utility.Vector3dVector(frame1_colors_norm)
# pcd.colors = colors

# # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.001)
# # o3d.visualization.draw_geometries([voxel_grid])

# # Create a coordinate frame: x-axis -> red, y-axis -> green, z-axis -> blue
# coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# octree = o3d.geometry.Octree(max_depth=4)
# octree.convert_from_point_cloud(pcd, size_expand=0.01)
# o3d.visualization.draw_geometries([octree, coordinate_frame])