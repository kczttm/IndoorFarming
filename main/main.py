# Alex Qiu - Main Script for RAFT Using https://github.com/itberrios/CV_projects/tree/main/RAFT Source Code
# Needs NVIDIA GPU to run (CUDA-enabled GPU)

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
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
from helper_functions import *

print("Is torch.cuda Available? ", torch.cuda.is_available())
print("Device Count: ",torch.cuda.device_count())
print("Current Device: ", torch.cuda.device(torch.cuda.current_device()))
print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

# Add RAFT core to path
sys.path.append('RAFT/core')

# Load model
model = load_model("models/raft-sintel.pth", args=Args())

# Select images to test model on
frame1 = cv2.imread('Test_Leaf_Obstacle_Frame_2.png')
frame2 = cv2.imread('Test_Leaf_Obstacle_Frame_3.png')

# Convert images to rgb color space
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

# Resize the second image to match the dimensions of the first image
height, width, channels = frame1.shape
frame2 = cv2.resize(frame2, (width, height))

# time1 = time.perf_counter()

# # Perform initial inference on model. Change device to cuda if available
# flow_low, flow_up = inference(model, frame1, frame2, device='cpu') # Flow low is low resolution version from "test mode", flow up is estimate after iterative refinement process
# flow_low.shape, flow_up.shape
# flow_low_viz = get_viz(flow_low)
# flow_up_viz = get_viz(flow_up)
# time2 = time.perf_counter()
# print("Inference Time: ", time2 - time1)
# f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20,10))

# # Display plots
# ax0.imshow(frame1)
# ax0.set_title("Frame 1 Original")
# ax0.set_xlabel("Pixels"); ax0.set_ylabel("Pixels")
# ax1.imshow(flow_low_viz)
# ax1.set_title("Low Resolution Optical Flow Visualization")
# ax1.set_xlabel("Pixels"); ax1.set_ylabel("Pixels")
# ax2.imshow(flow_up_viz)
# ax2.set_title('High Resolution Optical Flow Visualization')
# ax2.set_xlabel("Pixels"); ax2.set_ylabel("Pixels")
# plt.show()

# Running model again but not in test mode
time1 = time.perf_counter()
flow_iters = inference(model, frame1, frame2, device='cpu', iters=20, test_mode=False)
time2 = time.perf_counter()
print("Inference Time: ", time2 - time1)

# Plotting comparison between first and last iteration
fig, axes = plt.subplots(3, 2, figsize=(20, 16), gridspec_kw={'height_ratios': [4, 4, 1]})
axes[0, 0].imshow(frame1)
axes[0, 0].set_title("Frame 1")
axes[0, 1].imshow(frame2)
axes[0, 1].set_title("Frame 2")

im0 = axes[1, 0].imshow(get_viz(flow_iters[0]))
axes[1, 0].set_title('First RAFT Flow Iteration')
axes[1, 0].set_xlabel("Pixels"); axes[1, 0].set_ylabel("Pixels")
im1 = axes[1, 1].imshow(get_viz(flow_iters[-1]))
axes[1, 1].set_title('Final RAFT Flow Results')
axes[1, 1].set_xlabel("Pixels"); axes[1, 1].set_ylabel("Pixels")

# Add color bars
axes[2, 0].axis('off')
axes[2, 1].axis('off')
cbar0 = fig.colorbar(im0, ax=axes[2, :], fraction=0.4, pad=0.04, location="top")
cbar0.set_label('Magnitude of Optical Flow')
cbar0.set_ticks([0, 250])

plt.show()

final_flow = flow_iters[-1]
print("Final Flow Tensor Shape: ", final_flow.shape)
flow_x_matrix = final_flow[0, 0] # Create matrix of just horizontal displacement component (shape should be H X W of image plus padding)
flow_y_matrix = final_flow[0, 1] # Create matrix of just vertical displacement component (shape should be H X W of image plus padding)
flow_magnitude = torch.sqrt(flow_x_matrix**2 + flow_y_matrix**2)  # Compute magnitude using Euclidean norm

# Now compute the gradient of the flow magnitude
flow_magnitude.requires_grad = True  # Enable gradient computation
gradient = torch.autograd.grad(outputs=flow_magnitude.sum(), inputs=flow_magnitude)[0]
pd.options.display.max_rows = 4000
from IPython.display import display

# Set options to prevent truncation
display.max_columns = None  # Show all columns
display.max_rows = None     # Show all rows
print(gradient)