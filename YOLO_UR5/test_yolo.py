import sys
sys.path.append('../') # To call files from parent directory 
import time
importTimerStart = time.perf_counter()
totalTimerStart = time.perf_counter()
import socket
import cv2
import numpy as np
from yolov5.detect import yolo_detection
import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                            increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from Alex_Full_Stereo_V2 import apply_yolo_to_image
importTimerFinish = time.perf_counter()
print("\nTotal Time to Import Related Dependencies: " + str(importTimerFinish - importTimerStart) + "seconds \n")

YOLO_WEIGHT_FILE = "./yolov5/runs/train/exp13/weights/best.pt"
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

annotatedImage, centroids = apply_yolo_to_image('1_Berry_5mm.png')
print(centroids)


