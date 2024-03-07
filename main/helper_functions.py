import torch
import numpy as np
import sys
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
sys.path.append('../RAFT/core') # Add to path

if torch.cuda.is_available():
    curr_device = 'cuda'
    print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    curr_device = 'cpu'

def process_img(img, device):
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)

def load_model(weights_path, args):
    model = RAFT(args)
    print("hiiiii")
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
    baseline_x = 0.005
    tilt = np.radians(0) 
    focal_length = 891.77161 
    cam_x_offset = 0.000 
    cam_y_offset = 0.000
    pitch_angle = np.radians(0) 

    # Calculations
    world_conversion = baseline_x * np.cos(tilt) / flow_x_matrix # Calculating pixel to world conversion
    x_offsets = cam_x_offset - baseline_x # Including camera offset from center of gripper (if using a horizontal offset)
    x_translation_matrix = world_conversion * (image_center_x - pixel_x) # Calculating x translation using world conversion

    y_offsets = cam_y_offset
    pitch_offset = -((baseline_x * focal_length) / flow_x_matrix) * np.sin(pitch_angle) # Essentially using z-depth to find pitch offset
    y_translation_matrix = -world_conversion * (image_center_y - pixel_y) # Negated because camera is flipped upside down to match camera axes

    z_coord = ((baseline_x * focal_length) / -flow_x_matrix) # 2D matrix of depth values for each pixel in image
    # print(z_coord)
    x_coord = x_offsets - x_translation_matrix # 2D matrix of x-coordinate values for each pixel in image
    y_coord = pitch_offset + y_offsets + y_translation_matrix # 2D matrix of y-coordinate values for each pixel in image

    # Tuning matrix by getting rid of negative depths and zeros and make them infinity
    z_coord = np.where(z_coord <= 0, np.inf, z_coord) 

    return x_coord, y_coord, z_coord