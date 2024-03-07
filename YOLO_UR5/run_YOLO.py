
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

import socket
import time
import cv2
import numpy as np
import time
# from yolov5.detect import yolo_detection
import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from Alex_Full_Stereo_V2 import *

if torch.cuda.is_available():
    mac = False
    curr_device = 'cuda'
    print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    # Windows path 
    YOLO_WEIGHT_FILE = "C:\\Users\\Alex\\OneDrive - Georgia Institute of Technology\\Documents\\Georgia_Tech\\GTRI FarmHand\\Code\\IndoorFarming\\YOLO_UR5\\best.pt"
else:
    curr_device = 'cpu'
    mac = True
    # Macbook path
    YOLO_WEIGHT_FILE = "/Users/alex/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Documents/Georgia_Tech/GTRI FarmHand/Code/IndoorFarming/YOLO_UR5/yolov5/runs/train/exp13/weights/best.pt"

IMAGE_CENTER = [640, 360]
max_intensity_threshold = 180
min_intensity_threshold = 100

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
THICKNESS = 2

# Color 
RED = (0, 0, 255) # OPENCV uses BGR not RGB
WHITE = (255, 255, 255)
MODEL = None

