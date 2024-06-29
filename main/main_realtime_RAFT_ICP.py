# source the ros2_kinova_ws/install/setup.bash before running this script
import os, sys
import cv2
import numpy as np

script_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
sys.path.append(repo_root)

from Strawberry_Plant_Detection.detect import detect_boxes_only
from gen3_7dof.take_pictures_action_client import main as TakePicturesActionClient
from RAFT_tool_box import load_model, inference, display_flow
from RAFT_tool_box import get_largest_flower_box, filter_flow, gen_3d_points

from ICP_tool_box import get_flower_template_pcd, draw_registration_result
from ICP_tool_box import preprocess_point_cloud, np_to_o3d_point_cloud
from ICP_tool_box import execute_global_registration, refine_registration


DEBUG = True  # use images from the repo instead of taking pictures

def take_pictures(spacing=0.005):
    pictures = TakePicturesActionClient(spacing=spacing)
    return pictures[0], pictures[-1]


def main():
    RAFT_model = load_model()
    if DEBUG:
        img1_name = "Flower_on_vine_1_5mm_offset.png"
        img2_name = "Flower_on_vine_2_5mm_offset.png"
        frame1 = cv2.imread(os.path.join(script_dir, img1_name))
        frame2 = cv2.imread(os.path.join(script_dir, img2_name))
    else:
        frame1, frame2 = take_pictures()
        # save the images
        # cv2.imwrite("frame1_low_light.png", frame1)
        # cv2.imwrite("frame2_low_light.png", frame2)

    flow_iters = inference(RAFT_model, frame1, frame2, iters=50, test_mode=False) 
    final_flow = flow_iters[-1]
    display_flow(final_flow)
    boxes = detect_boxes_only(frame1, confidence=0.7)
    flower_box = get_largest_flower_box(boxes)
    flow_x, flow_y, kept_idx = filter_flow(final_flow, flower_box, visualize=True)

    # obtain 3D points
    x_p, y_p, z_p = gen_3d_points(flow_x, flow_y, kept_idx)
    
    target_flower_3d_points = np.column_stack((x_p, y_p, z_p))
    print("Remaining target flower Points Shape: ", target_flower_3d_points.shape)

    target_flower_pcd = np_to_o3d_point_cloud(target_flower_3d_points)
    template_flower_pcd = get_flower_template_pcd(visualize=False)

    # Preprocess the point clouds
    voxel_size = 0.001
    source_down, source_fpfh = preprocess_point_cloud(template_flower_pcd, voxel_size=voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_flower_pcd, voxel_size=voxel_size)

    # RANSAC based registration
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    # Refine with ICP
    result_ICP = refine_registration(source_down, target_down, result_ransac.transformation, voxel_size)
    print("ICP Inlier_rmse: ", result_ICP.inlier_rmse)

    draw_registration_result(source_down, target_down, result_ICP.transformation)

if __name__ == '__main__':
    main()