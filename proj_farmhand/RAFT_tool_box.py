import torch
import cv2
import numpy as np
import os, sys
np.set_printoptions(suppress=True)

# Get the absolute path of the current script
repo_root = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
# RAFT_path = os.path.join(repo_root, 'RAFT')
model_path = os.path.join(repo_root, 'RAFT', 'models', 'raft-sintel.pth')
# sys.path.append(RAFT_path)
# print(RAFT_path)

from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder

if torch.cuda.is_available():
    curr_device = 'cuda'
    print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    curr_device = 'cpu'
    print("Device Name: CPU")


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


def process_img(img, device=curr_device):
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)

def load_model(weights_path=model_path, args=Args(), device=curr_device):
    model = RAFT(args)
    pretrained_weights = torch.load(
        weights_path, map_location=torch.device(device)) # Change to cuda if available
    model = torch.nn.DataParallel(model)
    model.load_state_dict(pretrained_weights)
    model.to(device) # Change to cuda if available
    return model

def inference(model, frame1, frame2, device=curr_device, pad_mode='sintel', iters=12, flow_init=None, upsample=True, test_mode=True):

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

def display_img_cv2(img, window_name="Image"):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_flow(flow):
    flow_viz = cv2.cvtColor(get_viz(flow), cv2.COLOR_RGB2BGR)
    cv2.imshow("Flow Visualization", flow_viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def get_largest_flower_box(boxes):
    max_area = 0
    flower_box = None

    for i in range(len(boxes.cls)):
        if boxes.cls[i] == 0:  # Assuming class label 0 represents "flower"
            box = boxes.xyxy[i]
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > max_area:
                max_area = area
                flower_box = box

    # convert to numpy array and round to integers
    flower_box = np.round(flower_box.cpu().numpy()).astype(int)

    # print("XYXY indices of the biggest flower box:", flower_box)
    return flower_box


def filter_flow(flow, flower_box, z_score_threshold=-0.3, visualize=True):
    flow_x = flow[0, 0]
    flow_y = flow[0, 1]
    # Compute the magnitude of flow
    flow_magnitude = torch.sqrt(flow_x**2 + flow_y**2)
    flower_flow_magnitude = flow_magnitude[flower_box[1]:flower_box[3],
                                        flower_box[0]:flower_box[2]]
    mean_flower_flow = torch.mean(flower_flow_magnitude)
    std_flower_flow = torch.std(flower_flow_magnitude)
    flow_threshold = mean_flower_flow + z_score_threshold * std_flower_flow
    print("Threshold: ", flow_threshold)

    keep_bool = flow_magnitude >= flow_threshold
    flower_mask = torch.zeros_like(flow_magnitude)
    flower_mask[flower_box[1]:flower_box[3], flower_box[0]:flower_box[2]] = 1

    if visualize:
        # visualize the filtered flow on cv2
        flow_filtered = torch.where(keep_bool, flow_magnitude, torch.tensor(0.0))
        flow_filtered_w_mask = torch.where(flower_mask == 1, flow_filtered, torch.tensor(0.0))
        display_img_cv2(flow_filtered_w_mask.cpu().numpy(), window_name="Filtered Flow")

    kept_indices = torch.where(keep_bool, torch.tensor(1), torch.tensor(0))
    kept_indices = torch.logical_and(kept_indices, flower_mask)

    removed_indices = torch.where(kept_indices == 0, torch.tensor(1), torch.tensor(0))
    flow_x[removed_indices] = torch.inf
    flow_y[removed_indices] = torch.inf
    
    return flow_x.cpu().numpy(), flow_y.cpu().numpy(), kept_indices.cpu().numpy()

    
# Converts pixel location (either matrix of pixels or single pixel) to tool frame coordinate system
def pix_to_pos(pixel_x, pixel_y, flow_x_matrix, image_center_x, image_center_y, baseline_x=0.005):
    # Fixed the bug that the 3d point cloud had the baseline offset included
    # by Chuizheng Kong on 06/28/2024

    # Declare stereo related variables
    tilt = np.radians(0); focal_length = 474.2788 
    cam_x_offset = 0; cam_y_offset = 0; pitch_angle = np.radians(0) 

    # Calculations
    world_conversion = baseline_x * np.cos(tilt) / flow_x_matrix # Calculating pixel to world conversion
    x_offsets = cam_x_offset # Kong: Here was the bug
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

    # rotate 180 degrees about z-axis to match camera frame
    x_coord = -x_coord
    y_coord = -y_coord

    return x_coord, y_coord, z_coord


def gen_3d_points(flow_x, flow_y, kept_indices, pic_spacing=0.005):
    # Get the image center
    image_center_x = flow_x.shape[1] / 2
    image_center_y = flow_x.shape[0] / 2

    # Generate 3D points
    x, y = np.meshgrid(np.arange(flow_x.shape[1]), np.arange(flow_x.shape[0]))

    x_coord, y_coord, z_coord = pix_to_pos(x, y, flow_x, image_center_x, image_center_y, baseline_x=pic_spacing)

    # filter inf values
    max_z_coord = np.round(np.max(z_coord[np.isfinite(z_coord)]), decimals=3)

    x_coord_filtered = np.where(z_coord <= max_z_coord, x_coord, 0)
    y_coord_filtered = np.where(z_coord <= max_z_coord, y_coord, 0)
    z_coord_filtered = np.where(z_coord <= max_z_coord, z_coord, 0)

    # filter out the points that are not in the flower box
    x_p = x_coord_filtered[kept_indices]
    y_p = y_coord_filtered[kept_indices]
    z_p = z_coord_filtered[kept_indices]

    return x_p, y_p, z_p


def flower_further_extraction():
    # use cv2.createBackgroundSubtractorMOG2() to extract the flower from the background
    pass