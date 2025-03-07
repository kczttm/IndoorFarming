# source the ros2_kinova_ws/install/setup.bash before running this script
import os, sys
import cv2
import numpy as np
import time

script_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
sys.path.append(repo_root)

from Strawberry_Plant_Detection.detect import detect_boxes_only

from RAFT_tool_box import load_model, inference, display_flow
from RAFT_tool_box import get_largest_flower_box, filter_flow, gen_3d_points

from ICP_tool_box import get_flower_template_pcd, draw_registration_result, rotate_pcd_htm
from ICP_tool_box import preprocess_point_cloud, np_to_o3d_point_cloud
from ICP_tool_box import execute_global_registration, refine_registration
from ICP_tool_box import rotation_matrix_to_euler, rotate_frame_on_ball

from gen3_7dof.take_pictures_action_client import main as TakePicturesActionClient
from gen3_7dof.tool_box import get_endoscope_tf_from_yaml, get_polli_fork_tf_from_yaml, tf_to_hom_mtx, H_mtx_to_kinova_pose_in_base
from gen3_7dof.tool_box import TCPArguments, move_tool_pose_absolute, move_tool_pose_relative, get_world_EE_HomoMtx
from gen3_7dof.utilities import DeviceConnection

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2



def take_pictures(spacing=0.005):
    pictures = TakePicturesActionClient(spacing=spacing)
    return pictures[0], pictures[-1]

def pose_estimation(visualize=False):
    RAFT_model = load_model()
    pic_spacing = 0.005

    frame1, frame2 = take_pictures(spacing=pic_spacing)
    # save the images
    # cv2.imwrite("frame1_low_light.png", frame1)
    # cv2.imwrite("frame2_low_light.png", frame2)

    flow_iters = inference(RAFT_model, frame1, frame2, iters=50, test_mode=False) 
    final_flow = flow_iters[-1]
    if visualize:
        display_flow(final_flow)
    boxes = detect_boxes_only(frame1, confidence=0.7)
    flower_box = get_largest_flower_box(boxes)
    flow_x, flow_y, kept_idx = filter_flow(final_flow, flower_box, visualize=False)

    # obtain 3D points
    x_p, y_p, z_p = gen_3d_points(flow_x, flow_y, kept_idx, pic_spacing=pic_spacing)
    
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
    # draw_registration_result(source_down, target_down, result_ransac.transformation)

    # Refine with ICP
    result_ICP = refine_registration(source_down, target_down, result_ransac.transformation, voxel_size)

    max_try = 200
    count = 0
    best_result = result_ICP
    while result_ICP.inlier_rmse > 0.00087 and count < max_try:
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        result_ICP = refine_registration(source_down, target_down, result_ransac.transformation, voxel_size)
        if result_ICP.inlier_rmse < best_result.inlier_rmse:
            best_result = result_ICP
        count += 1
    
    print("Total ICP Iterations: ", count)
    print("Best ICP Inlier_rmse: ", best_result.inlier_rmse)
    result_ICP = best_result

    if visualize:
        draw_registration_result(source_down, target_down, result_ICP.transformation)

    H_flower_in_endo = result_ICP.transformation
    print("Transformation Matrix: \n", H_flower_in_endo)
    return H_flower_in_endo, source_down, target_down


def main():
    H_flower_in_endo, template_pcd, flower_pcd = pose_estimation(visualize = False)
    ## Preparing to align the robot with the front face of the flower
    # Decompose Rotation into Euler Angles
    flower_roll, flower_pitch, flower_yaw = rotation_matrix_to_euler(H_flower_in_endo[:3, :3])
    print("Roll: ", np.degrees(flower_roll))
    print("Pitch: ", np.degrees(flower_pitch))
    print("Yaw: ", np.degrees(flower_yaw))

    # account for flipped flower pose estimation
    if abs(np.degrees(flower_roll)) > 90:
        flower_pitch = -np.sign(flower_roll)*flower_pitch
    
    # need to ignore the yaw but account for the sign
    flower_pitch = -np.sign(flower_yaw)*flower_pitch

    # draw_registration_result(template_pcd, flower_pcd, H_flower_in_endo)

    H_endo_des = rotate_frame_on_ball(H_flower_in_endo[:3,3], 0, flower_pitch, 0)
    # print("Desired Endoscope Pose: \n", H_endo_des)

    # Connect to the robot
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        # Make sure the arm is in Single Level Servoing mode (high-level mode)
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(base_servo_mode)

        EE_endo_tf = get_endoscope_tf_from_yaml()
        EE_polli_fork_tf = get_polli_fork_tf_from_yaml()

        H_wd_ee = get_world_EE_HomoMtx(base)
        init_pose = base.GetMeasuredCartesianPose()
        p_init_kinova = np.array([init_pose.x, init_pose.y, init_pose.z, init_pose.theta_x, init_pose.theta_y, init_pose.theta_z])
        print("Initial Pose: \n", p_init_kinova)

        # print("World to EE Pose: \n", H_wd_ee)
        H_wd_endo = H_wd_ee @ tf_to_hom_mtx(EE_endo_tf) # get the endoscope pose in world frame
        # print("Endoscope Pose in World Frame: \n", H_wd_endo)
        # # Get the flower origin in world frame
        # p_flower_origin_wd_frame = H_wd_endo[:3,:] @ H_flower_in_endo[:,3]

        H_wd_endo_des = H_wd_endo @ H_endo_des
        print("Desired Endoscope Pose in World Frame: \n", H_wd_endo_des)

        H_wd_ee_des = H_wd_endo_des @ np.linalg.inv(tf_to_hom_mtx(EE_endo_tf))
        p_world = H_wd_ee_des[:3,3]
        R_ee = H_wd_ee_des[:3,:3]

        r_wd, p_wd, y_wd = rotation_matrix_to_euler(R_ee)
        r_wd, p_wd, y_wd = np.degrees(r_wd), np.degrees(p_wd), np.degrees(y_wd)
        p_orient_kinova = np.array([p_world[0], p_world[1], p_world[2], r_wd, p_wd, y_wd])
        print("Desired Reorienting Pose: \n", p_orient_kinova)

        ## Get the bottom of the flower pose in the fork frame after reorienting
        H_wd_polli_fork_init = H_wd_ee_des @ tf_to_hom_mtx(EE_polli_fork_tf)
        H_polli_fork_endo_old = np.linalg.inv(H_wd_polli_fork_init) @ H_wd_endo

        # find the lowest point of the flower in the fork frame
        flower1_pcd_fork_frame = rotate_pcd_htm(np.asarray(template_pcd.points), H_polli_fork_endo_old @ H_flower_in_endo)

        bottom_idx = np.argmax(flower1_pcd_fork_frame[:,1])
        bottom_y = flower1_pcd_fork_frame[bottom_idx,1]
        bottom_z = flower1_pcd_fork_frame[bottom_idx,2]
        p_flower_origin_fork_frame = H_polli_fork_endo_old @ H_flower_in_endo[:,3]

        # print(bottom_y, bottom_z, p_flower_origin_fork_frame)
        extend_z = max(bottom_z, p_flower_origin_fork_frame[2])
        
        H_polli_fork_des = np.eye(4)
        H_polli_fork_des[0:3, 3] = [0.0, bottom_y+0.002, extend_z+0.005]
        H_wd_polli_fork_des = H_wd_polli_fork_init @ H_polli_fork_des
        p_polli_fork_des_kinova = H_mtx_to_kinova_pose_in_base(H_wd_polli_fork_des @ np.linalg.inv(tf_to_hom_mtx(EE_polli_fork_tf)))
        print("Desired Polli Fork Pose: \n", p_polli_fork_des_kinova)



        # ## drive the robot back 5cm
        # H_wd_polli_fork = get_world_EE_HomoMtx(base) @ tf_to_hom_mtx(EE_polli_fork_tf)
        # p_backword_wd = H_wd_polli_fork[:3,:3] @ np.array([0,0,-0.05])
        # pose_backword_kinova = np.array([p_backword_wd[0], p_backword_wd[1], p_backword_wd[2], 0, 0, 0])
        # action_result_2 = move_tool_pose_relative(base, base_cyclic, pose_backword_kinova)

        

        # ## drive the robot to touch the center of the flower
        # # get the current polli fork pose
        # H_wd_ee = get_world_EE_HomoMtx(base)
        # H_wd_polli_fork = H_wd_ee @ tf_to_hom_mtx(EE_polli_fork_tf)
        # p_polli_fork_origin_wd_frame = H_wd_polli_fork[:3,:] @ np.array([0,0,0.01,1])
        # p_diff_wd_frame = p_flower_origin_wd_frame - p_polli_fork_origin_wd_frame
        # pose_diff_kinova = np.array([p_diff_wd_frame[0], p_diff_wd_frame[1], p_diff_wd_frame[2], 0, 0, 0])
        # action_result_3 = move_tool_pose_relative(base, base_cyclic, pose_diff_kinova)
        
        # # time.sleep(40)
        # # request user input in the terminal, q to quit the python script
        # while True:
        #     user_input = input("Press q to confirm complete: ")
        #     if user_input == 'q':
        #         break



        ## drive the robot back to inital position
        action_result_1 = move_tool_pose_absolute(base, p_orient_kinova)
        action_result_2 = move_tool_pose_absolute(base, p_polli_fork_des_kinova)
        # request user input in the terminal, q to quit the python script
        while True:
            user_input = input("Press q to confirm complete: ")
            if user_input == 'q':
                break
        action_result_3 = move_tool_pose_absolute(base, p_orient_kinova)
        action_result_4 = move_tool_pose_absolute(base, p_init_kinova)

        




if __name__ == '__main__':
    main()