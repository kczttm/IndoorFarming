import os
import socket
import time
import sys
import numpy as np
import cv2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Suppress import warnings
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
import open3d as o3d

sys.path.append('/Users/alex/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Documents/Georgia_Tech/GTRI FarmHand/Code/IndoorFarming/Path_Planning') # Add to path
sys.path.append('/Users/alex/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Documents/Georgia_Tech/GTRI FarmHand/Code/IndoorFarming/YOLO_UR5') # Add to path
sys.path.append('../IndoorFarming/YOLO_UR5/utils')
sys.path.append('../IndoorFarming/Path_Planning') # Add to path
# Add the parent directory of 'main' to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname('main.py'), os.pardir))
sys.path.append(os.path.join(parent_dir, 'RAFT'))

from rrt_octree import *
from Alex_Full_Stereo_V2 import apply_yolo_to_image

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

print("Is torch.cuda available? ", torch.cuda.is_available())
print("Device Count: ",torch.cuda.device_count())

if torch.cuda.is_available():
    curr_device = 'cuda'
    print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    curr_device = 'cpu'

def process_img(img, device):
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)

def load_model(weights_path, args):
    model = RAFT(args)
    pretrained_weights = torch.load(
        weights_path, map_location=torch.device(curr_device)) # Change to cuda if available
    model = torch.nn.DataParallel(model)
    model.load_state_dict(pretrained_weights)
    model.to(curr_device) # Change to cuda if available
    return model

def inference(model, frame1, frame2, device, pad_mode='sintel', iters=12, flow_init=None, upsample=True, test_mode=True):

    model.eval()
    with torch.no_grad():
        # preprocess
        frame1 = process_img(frame1, device)
        frame2 = process_img(frame2, device)

        padder = InputPadder(frame1.shape, mode=pad_mode)
        frame1, frame2 = padder.pad(frame1, frame2)

        # predict flow
        if test_mode:
            flow_low, flow_up = model(frame1, frame2, iters=iters, flow_init=flow_init, upsample=upsample, test_mode=test_mode)
            return flow_low, flow_up

        else:
            flow_iters = model(frame1, frame2, iters=iters, flow_init=flow_init, upsample=upsample, test_mode=test_mode)
            return flow_iters

def get_viz(flo):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    return flow_viz.flow_to_image(flo)

# sketchy class to pass to RAFT
class Args():
    def __init__(self, model='', path='', small=False, mixed_precision=True, alternate_corr=False):
        self.model = model
        self.path = path
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr

    """ Sketchy hack to pretend to iterate through the class objects """

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration
    
# Converts pixel location (either matrix of pixels or single pixel) to tool frame coordinate system
def pix_to_pos(pixel_x, pixel_y, flow_x_matrix, image_center_x, image_center_y):

    # Declare stereo related variables
    baseline_x = 0.005; tilt = np.radians(0); focal_length = 891.77161 
    cam_x_offset = 0; cam_y_offset = 0; pitch_angle = np.radians(0) 

    # Calculations
    world_conversion = baseline_x * np.cos(tilt) / flow_x_matrix # Calculating pixel to world conversion
    x_offsets = cam_x_offset - baseline_x # Including camera offset from center of gripper (if using a horizontal offset)
    x_translation_matrix = world_conversion * (image_center_x - pixel_x) # Calculating x translation using world conversion

    y_offsets = cam_y_offset
    pitch_offset = -((baseline_x * focal_length) / flow_x_matrix) * np.sin(pitch_angle) # Essentially using z-depth to find pitch offset
    y_translation_matrix = -world_conversion * (image_center_y - pixel_y) # Negated because camera is flipped upside down to match camera axes

    z_coord = ((baseline_x * focal_length) / -flow_x_matrix) # 2D matrix of depth values for each pixel in image
    # print(z_coord)
    x_coord = -1 * (x_offsets - x_translation_matrix) # 2D matrix of x-coordinate values for each pixel in image
    y_coord = -1 * (pitch_offset + y_offsets + y_translation_matrix) # 2D matrix of y-coordinate values for each pixel in image

    # Tuning matrix by getting rid of negative depths and zeros and make them infinity
    z_coord = np.where(z_coord <= 0, np.inf, z_coord) 

    return x_coord, y_coord, z_coord

def build_pose_string(pose):
    # pose is a list with length of 6
    # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
    values = str(pose[0]) + ", " + str(pose[1]) + ", " + str(pose[2]) + \
        ", " + str(pose[3]) + ", " + str(pose[4]) + ", " + str(pose[5])
    poseString = "(" + values + ")"
    return poseString

def move_robot(pose):
    # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
    HOST = "169.254.14.134"  # Laptop (server) IP address.
    PORT = 30002

    # Communicate with UR5
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))  # Bind to the port.
    s.listen(5)  # Wait for UR5 (client) connection.
    c, addr = s.accept()  # Establish connection with client.

    try:
        msg = c.recv(1024).decode()  # Receive message from UR5.
        if msg == 'UR5_is_asking_for_data':
            poseString = build_pose_string(pose)
            c.send(poseString.encode())
            c.close()
            s.close()

    except socket.error as socketerror:
            print("Error occured.")

print("Successfully created helper functions.")

img1_name = "Test_Image_1_5mm_offset.png"
img2_name = "Test_Image_2_5mm_offset.png"

annotated_image_1, berry_centroids_1 = apply_yolo_to_image(img1_name)
annotated_image_2, berry_centroids_2 = apply_yolo_to_image(img2_name)

print("\nBerry Centroids 1: ", berry_centroids_1)
print("Berry Centroids 2: ", berry_centroids_2)

print("\nImage 1 Shape: ", annotated_image_1.shape)
print("Image 2 Shape: ", annotated_image_2.shape)

image_center_x = annotated_image_1.shape[1] / 2
image_center_y = annotated_image_1.shape[0] / 2

print("\nImage Center X: ", image_center_x)
print("Image Center Y: ", image_center_y)

frame_1_centroid_x = berry_centroids_1[:, 0]
frame_1_centroid_y = berry_centroids_1[:, 1]

print("\nFrame 1 Centroid X: ", frame_1_centroid_x)
print("Frame 1 Centroid Y: ", frame_1_centroid_y)

x_disparity = berry_centroids_2[:, 0] - berry_centroids_1[:, 0]

print("\nX Disparity: ", x_disparity)

# berry1_x = frame_1_centroid_x[0]
# berry1_y = frame_1_centroid_y[0]

berries_x_pos, berries_y_pos, berries_z_pos = pix_to_pos(frame_1_centroid_x, frame_1_centroid_y, x_disparity, image_center_x, image_center_y)
berries_pos_matrix = np.column_stack((berries_x_pos, berries_y_pos, berries_z_pos))
print("\nBerry Positions (m): ")
print(berries_pos_matrix)

fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(annotated_image_1)
axes[0].set_title('YOLO Berry Detection - Image 1')
axes[1].imshow(annotated_image_2)
axes[1].set_title('YOLO Berry Detection - Image 2')
# plt.tight_layout()
# plt.show()

model = load_model("../RAFT/models/raft-sintel.pth", args=Args())
frame1 = cv2.imread(img1_name)
frame2 = cv2.imread(img2_name)

# # Downsample image to reduce computation complexity (using Lanczos4 downsampling algorithm)
# frame1 = cv2.resize(frame1, (1512, 851), interpolation=cv2.INTER_LANCZOS4)
# frame2 = cv2.resize(frame2, (1512, 851), interpolation=cv2.INTER_LANCZOS4)

print(frame1.shape)

# Convert images to rgb color space
frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

# Resize the second image to match the dimensions of the first image
height, width, channels = frame2.shape
frame1 = cv2.resize(frame1_rgb, (width, height))

# Apply RAFT Optical Flow Model on Frame 1 and 2
time1 = time.perf_counter()
flow_iters = inference(model, frame1, frame2_rgb, device=curr_device, iters=25, test_mode=False)
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
# plt.show()

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

# np.savetxt('flow_x_matrix.txt', flow_x_matrix)
# np.savetxt('flow_y_matrix.txt', flow_y_matrix)
# np.savetxt('flow_magnitude.txt', flow_magnitude)

# Extra Information
max_index = np.unravel_index(np.argmax(flow_x_matrix, axis=None), flow_x_matrix.shape)
print("Index of maximum flow-x value:", max_index)
print("Max. Flow Magnitude: ", np.max(flow_magnitude))
baseline_x = 0.005
tilt = np.radians(0) 

print("Max. Flow in x-direction :", np.max(flow_x_matrix), "pixels")
print("Min. Flow in x-direction :", np.min(flow_x_matrix), "pixels")

print("Max. Flow in y-direction :", np.max(flow_y_matrix), "pixels")
print("Min. Flow in y-direction :", np.min(flow_y_matrix), "pixels")

# Calculating pixel to world conversion
world_conversion = baseline_x * np.cos(tilt) / np.max(flow_x_matrix) 
print("Pixels per mm at the depth where max disparity is: ", (1 / world_conversion) / 1000)

# Apply threshold to flow magnitude
flow_threshold = 33  # Pixel displacement threshold value
flow_magnitude_filtered = np.where(flow_magnitude < flow_threshold, 0, flow_magnitude) # New matrix of pixels that turn black (value set to 0) if not passing flow threshold
removed_indices = np.where(flow_magnitude_filtered == 0)

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
im_center_y = final_flow[0][0].shape[0] / 2 
x_pos_matrix, y_pos_matrix, z_pos_matrix = pix_to_pos(x_pix_matrix, y_pix_matrix, flow_x_matrix, im_center_x, im_center_y) 
kept_indices = np.where(np.isfinite(z_pos_matrix))

x_pos_filtered = x_pos_matrix[kept_indices]
y_pos_filtered = y_pos_matrix[kept_indices]
z_pos_filtered = z_pos_matrix[kept_indices]     
obstacle_3d_points = np.column_stack((x_pos_filtered, y_pos_filtered, z_pos_filtered))

print("Remaining Obstacle Points Shape: ", obstacle_3d_points.shape)

print("\nMaximum finite x value (m): ", np.round(np.max(x_pos_matrix[np.isfinite(x_pos_matrix)]), decimals=4))
print("Minimum x value (m): ", np.round(np.min(x_pos_matrix), decimals=4))

print("\nMaximum finite y value (m): ", np.round(np.max(y_pos_matrix[np.isfinite(y_pos_matrix)]), decimals=4))
print("Minimum y value (m): ", np.round(np.min(y_pos_matrix), decimals=4))

print("\nMaximum finite z value (m): ", np.round(np.max(z_pos_matrix[np.isfinite(z_pos_matrix)]), decimals=4))
print("Minimum depth value (m): ", np.round(np.min(z_pos_matrix), decimals=4))

x_p = x_pos_filtered
y_p = y_pos_filtered
z_p = z_pos_filtered

# Create 3D point cloud plot using plotly
fig_3d = px.scatter_3d(x=x_p, y=y_p, z=z_p, color=z_p, opacity=1, size_max=10, color_continuous_scale='Viridis')
fig_3d.update_layout(title='3D Point Cloud', coloraxis_colorbar=dict(title='Z_Depth Magnitude'), width=1000, height=500)

# Find the maximum range among the axes
max_range = max(max(x_p) - min(x_p), max(y_p) - min(y_p), max(z_p) - min(z_p))

# Calculate upper bounds
x_max = min(x_p) + max_range
y_max = min(y_p) + max_range
z_max = min(z_p) + max_range

# Set equal ranges for all axes
fig_3d.update_layout(scene=dict(xaxis=dict(range=[min(x_p), x_max]), yaxis=dict(range=[min(y_p), y_max]), zaxis=dict(range=[min(z_p), z_max]), aspectmode='cube'))
# fig_3d.show()

fig_2d_xz = px.scatter(x=x_pos_filtered, y=z_pos_filtered)
fig_2d_xz.update_layout(title='XZ View of Point Cloud')
fig_2d_xz.update_xaxes(title_text='X-position (m)')
fig_2d_xz.update_yaxes(title_text='Z-position (m)')
# fig_2d_xz.show()

fig_2d_xy = px.scatter(x=x_pos_filtered, y=y_pos_filtered)
fig_2d_xy.update_layout(title='XY View of Point Cloud')
fig_2d_xy.update_xaxes(title_text='X-position (m)')
fig_2d_xy.update_yaxes(title_text='Y-position (m)')
# fig_2d_xy.show()

pcd = o3d.geometry.PointCloud() # Init point cloud object
points = o3d.utility.Vector3dVector(obstacle_3d_points) # Convert numpy array of points to Open3D Vector
pcd.points = points # Assigning point cloud object field

frame1_colors = frame1[kept_indices] # Extracting pixels that are kept after threshold 
frame1_colors_norm = frame1_colors / 255.0 # Normalize color values to [0, 1]

colors = o3d.utility.Vector3dVector(frame1_colors_norm)
pcd.colors = colors

# Create Voxel Grid
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.001)
# o3d.visualization.draw_geometries([voxel_grid])

# Create a coordinate frame: x-axis -> red, y-axis -> green, z-axis -> blue
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

octree_max_depth = 6
octree = o3d.geometry.Octree(max_depth=octree_max_depth)
octree.convert_from_point_cloud(pcd, size_expand=0.01)

# Get Octree Bounding Box Parameters
bounding_box = octree.get_axis_aligned_bounding_box()
min_bound = bounding_box.get_min_bound()
max_bound = bounding_box.get_max_bound()
length = max_bound[0] - min_bound[0] 
width = max_bound[1] - min_bound[1]
height = max_bound[2] - min_bound[2]
diagonal_length = np.linalg.norm(np.array(max_bound) - np.array(min_bound))
voxel_size = length / (2 ** octree_max_depth) # Calculating size of minimum voxel size (this is correct, can check with OctreeNodeInfo)

print("Bounding Box Length: ", length)
print("Bounding Box Width: ", width)
print("Bounding Box Height: ", height)
print("Bounding Box Diagonal Length: ", diagonal_length)
print("Voxel Size Resolution: ", voxel_size)

# o3d.visualization.draw([octree, coordinate_frame])

# Set RRT parameters
goal_pos = np.array(berries_pos_matrix[0]) # Goal position
goal_pos[2] = goal_pos[2] + 0.07
goal_pos[0] = goal_pos[0] - 0.02
print("Goal Position: ", goal_pos)
# obstacle_list = np.array([octree]) # Create numpy array for octree(s)

# Run RRT path planning algorithm 
rrt_obj, path, path_points_UR5, iteration_count = rrt_main(goal_pos, octree) # Find path 
print("Path Length: ", len(path))
print("Iterations Run: ", iteration_count)

blackberry_model = o3d.geometry.TriangleMesh.create_sphere(radius=0.02) # Create blackberry model for visualization
blackberry_color_norm = [x / 255 for x in [113, 50, 250]] # Create normalized rgb color vector for purple
blackberry_model.paint_uniform_color(blackberry_color_norm) # Set uniform color value
blackberry_model.translate(goal_pos) # Translate blackberry to its position

robot_model = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Create robot model for visualization
robot_color_norm = [x / 255 for x in [0, 0, 255]]
robot_model.paint_uniform_color(robot_color_norm) # Set rgb color value (normalized)

# Create point cloud of generated path from RRT for visualization
path_pointcloud = o3d.geometry.PointCloud()
path_pointcloud.points = o3d.utility.Vector3dVector(path)

# Create line segments connecting the path points
lines = []
for i in range(len(path) - 1):
    lines.append([i, i+1])

# Create an Open3D line set from the line segments
path_segments = o3d.geometry.LineSet()
path_segments.points = o3d.utility.Vector3dVector(path)
path_segments.lines = o3d.utility.Vector2iVector(lines)
path_color_norm = [x / 255 for x in [255, 0, 0]]
path_segments.paint_uniform_color(path_color_norm)

# Visualize workspace
# o3d.visualization.draw([octree, coordinate_frame, blackberry_model, robot_model, path_pointcloud, path_segments])

previous_point = None

time.sleep(5)

# Iterate through each point in path and calculate the translation command for UR5
for current_point in path_points_UR5:

    if previous_point is not None:
        movement_vector = current_point - previous_point
        # print("\nMoving robot: ", movement_vector)
        move_robot(movement_vector) # Move UR5 robot
        time.sleep(0.1)

    previous_point = current_point