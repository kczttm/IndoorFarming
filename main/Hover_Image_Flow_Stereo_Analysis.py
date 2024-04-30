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
import plotly.express as px
from collections import OrderedDict

sys.path.append('..IndoorFarming/RAFT/core') # Add to path
sys.path.append('/Users/alex/Documents/Georgia Tech/GTRI FarmHand/Code/IndoorFarming/YOLO_UR5') # Add to path
sys.path.append('../IndoorFarming/YOLO_UR5/utils')
sys.path.append('/Users/alex/Documents/Georgia Tech/GTRI FarmHand/Code/IndoorFarming')

# Add the parent directory of 'main' to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname('main.ipynb'), os.pardir))
sys.path.append(os.path.join(parent_dir, 'RAFT'))

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
from Strawberry_Plant_Detection import detect

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
    x_coord = (x_offsets - x_translation_matrix) # 2D matrix of x-coordinate values for each pixel in image
    y_coord = (pitch_offset + y_offsets + y_translation_matrix) # 2D matrix of y-coordinate values for each pixel in image

    # Tuning matrix by getting rid of negative depths and zeros and make them infinity
    # z_coord = np.where(z_coord <= 0, np.inf, z_coord) 

    return x_coord, y_coord, z_coord

img1_name = "Flower_Out_of_Page_1_5mm_offset.png"
img2_name = "Flower_Out_of_Page_2_5mm_offset.png"

model = load_model("../RAFT/models/raft-sintel.pth", args=Args()) # sintel may not always be the best model, try other ones if getting weird results
frame1 = cv2.imread(img1_name)
frame2 = cv2.imread(img2_name)

print("Image Resolution: ", frame1.shape)

# Convert images to rgb color space (KEEP AS BGR IF NEEDED, THIS DOES AFFECT RESULTS)
frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

# Apply RAFT Optical Flow Model on Frame 1 and 2
time1 = time.perf_counter()
flow_iters = inference(model, frame1, frame2, device=curr_device, iters=20, test_mode=False) 
time2 = time.perf_counter()
print("Inference Time: ", time2 - time1)

fig2, axes = plt.subplots(3, 2, figsize=(18, 12), gridspec_kw={'height_ratios': [4, 4, 1]})
fig2.subplots_adjust(hspace=0.3)
axes[0, 0].imshow(frame1_rgb)
axes[0, 0].set_title("Frame 1")
axes[0, 1].imshow(frame2_rgb)
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
mean_flow = np.mean(np.abs(flow_x_matrix))
print("Mean flow: ", mean_flow)
flow_threshold = mean_flow  # Pixel displacement threshold value
flow_filtered = np.where(np.abs(flow_x_matrix) < flow_threshold, 0, flow_x_matrix) # New matrix of pixels that turn black (value set to 0) if not passing flow threshold

removed_indices = np.where(np.abs(flow_x_matrix) < flow_threshold)
# print(len(removed_indices[0]))
kept_indices = np.where(np.abs(flow_x_matrix) >= flow_threshold)
# print(len(kept_indices[0]))

# Visualize the thresholded flow magnitude
fig3, axes = plt.subplots(1, 2, figsize=(18, 12))
fig3.subplots_adjust(hspace=0.3)
axes[0].imshow(frame1_rgb); axes[0].set_title("Frame 1"); axes[1].imshow(flow_filtered, cmap='gray'); axes[1].set_title("Segmented Flow")

# Adjust flow matrices so that the filtered out pixels are set to infinity so they can be ignored for later processing
flow_x_matrix[removed_indices] = np.inf
flow_y_matrix[removed_indices] = np.inf

# Creating matrix of pixels
height, width, channels = final_flow_viz.get_array().shape

# Create arrays representing the x and y coordinates of pixels
y_coords, x_coords = np.mgrid[0:height, 0:width]

# Stack x_coords and y_coords to create the matrix of pixels
pixel_matrix = np.dstack((x_coords, y_coords))
x_pix_matrix = pixel_matrix[:, :, 0]
y_pix_matrix = pixel_matrix[:, :, 1]

# Get matrices of (x, y, z) locations for each pixel in image
im_center_x = final_flow[0][0].shape[1] / 2
im_center_y = final_flow[0][0].shape[0] / 2 

# Calculate coordinates of each pixel in world frame
x_pos_matrix, y_pos_matrix, z_pos_matrix = pix_to_pos(x_pix_matrix, y_pix_matrix, flow_x_matrix, im_center_x, im_center_y) 

# kept_indices = np.where(np.isfinite(z_pos_matrix))
x_pos_filtered = np.where(z_pos_matrix <= 1, x_pos_matrix, 0)
y_pos_filtered = np.where(z_pos_matrix <= 1, y_pos_matrix, 0)
z_pos_filtered = np.where(z_pos_matrix <= 1, z_pos_matrix, 0)

x_pos_filtered = x_pos_filtered[kept_indices]
y_pos_filtered = y_pos_filtered[kept_indices]
z_pos_filtered = z_pos_filtered[kept_indices]  

obstacle_3d_points = np.column_stack((x_pos_filtered, y_pos_filtered, z_pos_filtered))

print("Remaining Obstacle Points Shape: ", obstacle_3d_points.shape)

print("\nMaximum finite x value (m): ", np.round(np.max(x_pos_matrix[np.isfinite(x_pos_matrix)]), decimals=4))
print("Minimum x value (m): ", np.round(np.min(x_pos_matrix), decimals=4))

print("\nMaximum finite y value (m): ", np.round(np.max(y_pos_matrix[np.isfinite(y_pos_matrix)]), decimals=4))
print("Minimum y value (m): ", np.round(np.min(y_pos_matrix), decimals=4))

print("\nMaximum finite z value (m): ", np.round(np.max(z_pos_matrix[np.isfinite(z_pos_matrix)]), decimals=4))
print("Minimum depth value (m): ", np.round(np.min(z_pos_matrix), decimals=4))


def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_MOUSEMOVE:
        # Retrieve pixel coordinates
        pixel_x, pixel_y = x, y
        # Retrieve flow displacement
        flow_displacement_x = flow_x_matrix[y, x]
        flow_displacement_y = flow_y_matrix[y, x]
        # Calculate world coordinates using pix_to_pos function
        world_x, world_y, world_z = pix_to_pos(pixel_x, pixel_y, flow_displacement_x, im_center_x, im_center_y)
        # Output the information
        print(f"Pixel: ({pixel_x}, {pixel_y})")
        print(f"Flow: ({np.round(flow_displacement_x, decimals=4)}, {np.round(flow_displacement_y, decimals=4)})")
        print(f"World: ({np.round(world_x, decimals=4)}, {np.round(world_y, decimals=4)}, {np.round(world_z, decimals=4)})")

# Create a window and set mouse callback
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

# Main loop for displaying the image and handling mouse events
while True:
    # Display the image
    cv2.imshow('image', frame1)
    cv2.resizeWindow('image', 800, 800)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cv2.destroyAllWindows()