# source the ros2_kinova_ws/install/setup.bash before running this script
import os, sys
import cv2
import numpy as np
import time

script_dir = os.path.dirname(__file__)
# repo_root = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
# sys.path.append(repo_root)

from Strawberry_Plant_Detection.detect import detect_boxes_only

from proj_farmhand.yolo_pursuit_action_client import main as YoloPursuitActionClient
from proj_farmhand.RAFT_tool_box import load_model, inference, display_flow
from proj_farmhand.RAFT_tool_box import get_largest_flower_box, filter_flow, gen_3d_points

from proj_farmhand.ICP_tool_box import get_flower_template_pcd, draw_registration_result, rotate_pcd_htm
from proj_farmhand.ICP_tool_box import preprocess_point_cloud, np_to_o3d_point_cloud
from proj_farmhand.ICP_tool_box import execute_global_registration, refine_registration
from proj_farmhand.ICP_tool_box import rotation_matrix_to_euler, rotate_frame_on_ball

from proj_farmhand.arduino_interfaces_tool_box import arduino_connect, auto_focus

from gen3_7dof.take_pictures_action_client import main as TakePicturesActionClient
from gen3_7dof.tool_box import euler_to_rotation_matrix
from gen3_7dof.tool_box import get_endoscope_tf_from_yaml, get_polli_fork_tf_from_yaml, tf_to_hom_mtx, H_mtx_to_kinova_pose_in_base
from gen3_7dof.tool_box import TCPArguments, move_tool_pose_absolute, move_tool_pose_relative, get_world_EE_HomoMtx
from gen3_7dof.utilities import DeviceConnection

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2


def robot_move_to_flower(percent_frame_height = 0.9):
    YoloPursuitActionClient(percent_frame_height=percent_frame_height)

def robot_take_pictures(spacing=0.005):
    pictures = TakePicturesActionClient(spacing=spacing)
    return pictures[0], pictures[-1]

def robot_pose_estimation(visualize=False, real_flower=False):
    RAFT_model = load_model()
    pic_spacing = 0.005

    frame1, frame2 = robot_take_pictures(spacing=pic_spacing)
    # save the images
    # cv2.imwrite("frame1_low_light.png", frame1)
    # cv2.imwrite("frame2_low_light.png", frame2)

    flow_iters = inference(RAFT_model, frame1, frame2, iters=50, test_mode=False) 
    final_flow = flow_iters[-1]
    if visualize:
        display_flow(final_flow)
    boxes = detect_boxes_only(frame1)
    flower_box = get_largest_flower_box(boxes)
    flow_x, flow_y, kept_idx = filter_flow(final_flow, flower_box, visualize=False)

    # obtain 3D points
    x_p, y_p, z_p = gen_3d_points(flow_x, flow_y, kept_idx, pic_spacing=pic_spacing)
    
    target_flower_3d_points = np.column_stack((x_p, y_p, z_p))
    print("Remaining target flower Points Shape: ", target_flower_3d_points.shape)

    target_flower_pcd = np_to_o3d_point_cloud(target_flower_3d_points)
    template_flower_pcd = get_flower_template_pcd(visualize=False, real_flower=real_flower)

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


def robot_move_in_endoscope_frame_relative(H_endo_des, speed=None):
    # note that the H_endo_des mapes between current endoscope pose and desired endoscope pose
    # so the input is NOT in world frame
    EE_endo_tf = get_endoscope_tf_from_yaml()
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)
        # Make sure the arm is in Single Level Servoing mode (high-level mode)
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(base_servo_mode)

        H_wd_ee = get_world_EE_HomoMtx(base)
        H_wd_endo = H_wd_ee @ tf_to_hom_mtx(EE_endo_tf) # get the endoscope pose in world frame
        # print("Endoscope Pose in World Frame: \n", H_wd_endo)

        H_wd_endo_des = H_wd_endo @ H_endo_des
        # print("Desired Endoscope Pose in World Frame: \n", H_wd_endo_des)

        H_wd_ee_des = H_wd_endo_des @ np.linalg.inv(tf_to_hom_mtx(EE_endo_tf))
        p_world = H_wd_ee_des[:3,3]
        R_ee = H_wd_ee_des[:3,:3]

        r_wd, p_wd, y_wd = rotation_matrix_to_euler(R_ee)
        r_wd, p_wd, y_wd = np.degrees(r_wd), np.degrees(p_wd), np.degrees(y_wd)
        p_des_kinova = np.array([p_world[0], p_world[1], p_world[2], r_wd, p_wd, y_wd])
        print("Desired Reorienting Pose: \n", p_des_kinova)

        action_result = move_tool_pose_absolute(base, p_des_kinova, speed=speed)
        return H_wd_ee_des, p_des_kinova


def robot_move_in_polli_fork_frame_absolute(H_wd_polli_fork_des, speed=None):
    EE_polli_fork_tf = get_polli_fork_tf_from_yaml()
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)
        # Make sure the arm is in Single Level Servoing mode (high-level mode)
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(base_servo_mode)

        H_wd_ee_des = H_wd_polli_fork_des @ np.linalg.inv(tf_to_hom_mtx(EE_polli_fork_tf))
        p_polli_fork_des_kinova = H_mtx_to_kinova_pose_in_base(H_wd_ee_des)
        action_result = move_tool_pose_absolute(base, p_polli_fork_des_kinova, speed=speed)
        return H_wd_ee_des, p_polli_fork_des_kinova


def get_current_EE_pose():
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)
        H_wd_ee = get_world_EE_HomoMtx(base)
        curr_pose = base.GetMeasuredCartesianPose()
        p_curr_kinova = np.array([curr_pose.x, curr_pose.y, curr_pose.z, curr_pose.theta_x, curr_pose.theta_y, curr_pose.theta_z])
        return H_wd_ee, p_curr_kinova
    

def robot_move_kinova_pose_series(p_kinova_series, velocity_series):
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)
        for i in range(len(p_kinova_series)):
            action_result = move_tool_pose_absolute(base, p_kinova_series[i], speed=velocity_series[i])
    return action_result

def cv2_video_display():
    cap = cv2.VideoCapture(4,cv2.CAP_V4L2)
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    REAL_FLOWER = True
    if REAL_FLOWER:
        per_H = 0.5
    else:
        per_H = 0.80

    SerialObj = arduino_connect()

    EE_endo_tf = get_endoscope_tf_from_yaml()
    EE_polli_fork_tf = get_polli_fork_tf_from_yaml()

    H_wd_ee, p_init_kinova = get_current_EE_pose()
    print("Initial Pose: \n", p_init_kinova)

    ################# Move to the flower #################
    robot_move_to_flower(percent_frame_height = per_H)
    H_wd_ee_curr, p_curr_kinova = get_current_EE_pose()
    print("Yolo Pursuit Pose: \n", p_curr_kinova)
    H_wd_endo_yolo = H_wd_ee_curr @ tf_to_hom_mtx(EE_endo_tf)

    ################# Align to the flower Stem #################
    H_flower_in_endo, template_pcd, flower_pcd = robot_pose_estimation(visualize = False, real_flower = REAL_FLOWER)
    ## Preparing to align the robot with the front face of the flower

    z_flower = H_flower_in_endo[:3, 2]
    flower_pitch = np.arctan2(z_flower[0], z_flower[2])
    print("Flower axis: ", z_flower)
    print("Flower Pitch: ", np.degrees(flower_pitch))
    # account for flipped flower pose estimation
    if z_flower[-1] < 0:
        flower_pitch = flower_pitch - np.pi
    print("Pitch: ", np.degrees(flower_pitch))
    
    draw_registration_result(template_pcd, flower_pcd, H_flower_in_endo)

    H_endo_des = rotate_frame_on_ball(H_flower_in_endo[:3,3], 0, flower_pitch, 0)
    print("Desired Endoscope Pose: \n", H_endo_des)

    H_wd_ee_des, p_orient_kinova = robot_move_in_endoscope_frame_relative(H_endo_des)
    # print("Desired Reorienting Pose: \n", p_orient_kinova)
    H_wd_ee_curr, p_curr_kinova = get_current_EE_pose()
    # print("Reorienting Pose error: ", p_orient_kinova - p_curr_kinova)


    # # draw_registration_result(template_pcd, flower_pcd, H_flower_in_endo)

    ################# Get to the bottom of the flower slowly #################
    ## Get the bottom of the flower pose in the fork frame after reorienting
    fork_depth = 0.005  # the depth of the fork in the flower (0.007 mm) z is pointing out
    fork_lower = 0.001 # the raise of the fork from the lowest point of the flower (0.001 mm) y is pointing down
    fork_raise = -0.001 # the raise of the fork from the center of the flower (-0.002 mm) y is pointing down
    fork_angle = 5.0 # the angle of the fork from the center of the flower (5 degrees) x is pointing to the right

    # map the post-yolo endoscope frame to the post-reorienting polli_fork frame
    H_wd_polli_fork_init = H_wd_ee_curr @ tf_to_hom_mtx(EE_polli_fork_tf)
    H_polli_fork_endo_yolo = np.linalg.inv(H_wd_polli_fork_init) @ H_wd_endo_yolo

    # find the lowest point of the flower in the fork frame
    flower1_pcd_fork_frame = rotate_pcd_htm(np.asarray(template_pcd.points), H_polli_fork_endo_yolo @ H_flower_in_endo)

    bottom_idx = np.argmax(flower1_pcd_fork_frame[:,1])
    bottom_y = flower1_pcd_fork_frame[bottom_idx,1]
    bottom_z = flower1_pcd_fork_frame[bottom_idx,2]
    p_flower_origin_fork_frame = H_polli_fork_endo_yolo @ H_flower_in_endo[:,3]

    # print(bottom_y, bottom_z, p_flower_origin_fork_frame)
    extend_z = max(bottom_z, p_flower_origin_fork_frame[2])
    
    H_polli_fork_des = np.eye(4)
    H_polli_fork_des[0:3, 3] = [p_flower_origin_fork_frame[0], bottom_y+fork_lower, extend_z+fork_depth]
    H_wd_polli_fork_des = H_wd_polli_fork_init @ H_polli_fork_des
    H_wd_ee_des, p_polli_fork_des_kinova = robot_move_in_polli_fork_frame_absolute(H_wd_polli_fork_des, speed=0.03)
    print("Desired Polli Fork Pose: \n", p_polli_fork_des_kinova)
    H_wd_ee_curr, p_curr_kinova = get_current_EE_pose()
    print("Move to bottom Pose error: ", p_polli_fork_des_kinova - p_curr_kinova)


    ################# raise to the center of the flower slowly #################
    H_polli_fork_des[0:3, 3] = [p_flower_origin_fork_frame[0], p_flower_origin_fork_frame[1]+fork_raise, extend_z+fork_depth]
    H_wd_polli_fork_des = H_wd_polli_fork_init @ H_polli_fork_des
    H_wd_ee_des, p_polli_fork_des_center_kinova = robot_move_in_polli_fork_frame_absolute(H_wd_polli_fork_des, speed=0.01)
    print("Desired Polli Fork Pose: \n", p_polli_fork_des_center_kinova)
    H_wd_ee_curr, p_curr_kinova = get_current_EE_pose()
    print("Raise to center Pose error: ", p_polli_fork_des_center_kinova - p_curr_kinova)


    ################# rotate +x of the fork frame and move -y #################
    H_polli_fork_des = np.eye(4)
    H_polli_fork_des[:3,:3] = euler_to_rotation_matrix(np.radians(fork_angle), 0, 0)
    H_polli_fork_des[0:3, 3] = [0.0, fork_raise, 0]
    H_polli_fork_curr = H_wd_ee_curr @ tf_to_hom_mtx(EE_polli_fork_tf)
    H_wd_polli_fork_des = H_polli_fork_curr @ H_polli_fork_des
    H_wd_ee_des, p_polli_fork_des_rotated_kinova = robot_move_in_polli_fork_frame_absolute(H_wd_polli_fork_des, speed=0.08)
    print("Desired Polli Fork Pose: \n", p_polli_fork_des_rotated_kinova)
    H_wd_ee_curr, p_curr_kinova = get_current_EE_pose()
    print("Raise to center Pose error: ", p_polli_fork_des_rotated_kinova - p_curr_kinova)

    ################# Keep moving slowly based on the vibration #################
    # try:
    #     cv2_video_display()
    # except KeyboardInterrupt:
    #     pass
    auto_focus(SerialObj)
    # input("Press Enter to continue...")

    ################# Return to the starting pose #################
    p_kinova_series = [p_polli_fork_des_kinova, p_orient_kinova, p_init_kinova]
    velocity_series = [0.05, None, None]  # None means default speed


    robot_move_kinova_pose_series(p_kinova_series, velocity_series)
        

if __name__ == '__main__':
    # main()
    get_flower_template_pcd(visualize=True, real_flower=True)