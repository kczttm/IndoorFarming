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

from proj_farmhand.arduino_interfaces_tool_box import arduino_connect, auto_focus, motor_command

from proj_farmhand.rs_sahi_global_flower_pose_client import main as RealSenseFlowerPosesActionClient

from proj_farmhand.flower_center_tracking import get_flower_center, calc_xy_plane_pose_error

from gen3_7dof.take_pictures_action_client import main as TakePicturesActionClient
from gen3_7dof.tool_box import get_endoscope_tf_from_yaml, get_polli_fork_tf_from_yaml, tf_to_hom_mtx, H_mtx_to_kinova_pose_in_base
from gen3_7dof.tool_box import get_realsense_on_link1_HomoMtx, euler_to_rotation_matrix, getRotMtx
from gen3_7dof.tool_box import TCPArguments, move_tool_pose_absolute, get_world_EE_HomoMtx, get_joint_angles, move_joints, move_end_effector_vel
from gen3_7dof.utilities import DeviceConnection

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

def robot_move_to_flower(percent_frame_height = 0.9):
    YoloPursuitActionClient(percent_frame_height=percent_frame_height)

def robot_take_pictures(spacing=0.005):
    pictures = TakePicturesActionClient(spacing=spacing)
    return pictures[0], pictures[-1]

def realsense_get_flower_poses(sahi_n_slices = 2):
    # returns flower poses in the camera frame (n-by-3 numpy array)
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(base_servo_mode)

        # get current joint angles
        joint_angles_init = get_joint_angles(base)
        try:
            H_wd_rs = get_realsense_on_link1_HomoMtx(base)
            print("Realsense Pose in World Frame: \n", H_wd_rs)
            # take pictures
            flower_poses_wd = RealSenseFlowerPosesActionClient(sahi_n_slices=sahi_n_slices)

            # trim the outliers
            flower_poses_wd = flower_poses_wd[flower_poses_wd[:,0] < 2]
            # sort the flower poses by depth
            flower_poses_wd = flower_poses_wd[flower_poses_wd[:,2].argsort()[::-1]]
        except:
            flower_poses_wd = None 

    return flower_poses_wd, joint_angles_init, H_wd_rs

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
        action_result = move_tool_pose_absolute(base, p_des_kinova, speed=speed)
        return H_wd_ee_des, p_des_kinova
    

def robot_move_in_endoscope_frame_absolute(H_wd_endo_des, speed=None):
    EE_endo_tf = get_endoscope_tf_from_yaml()
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)
        # Make sure the arm is in Single Level Servoing mode (high-level mode)
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(base_servo_mode)

        H_wd_ee_des = H_wd_endo_des @ np.linalg.inv(tf_to_hom_mtx(EE_endo_tf))
        p_des_kinova = H_mtx_to_kinova_pose_in_base(H_wd_ee_des)
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
    

def get_current_RS_pose():
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)
        H_wd_rs = get_realsense_on_link1_HomoMtx(base)
        return H_wd_rs
    

def robot_move_kinova_pose_series(p_kinova_series, velocity_series):
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)
        for i in range(len(p_kinova_series)):
            action_result = move_tool_pose_absolute(base, p_kinova_series[i], speed=velocity_series[i])
    return action_result


def robot_micro_adjust(SerialObj, REAL_FLOWER=False):
    # auth_radius is the maximum distance the robot could travel to explore.
    # align the flower to a good pos under the microscope
    # TODO if there are flowers that are facing downwards too much, hardcode a tilt angle till microscope sees it.

    # adjust the focus of the microscope
    motor_command(SerialObj, 'P', 635)
    motor_command(SerialObj, 'Z', 21)

    # create a list of centers and radii of the flowers to serve as a filtered list
    center_list = []
    ellipse_rad_list = []
    list_length = 5

    focal_length = 800 # in pixels, estimated, need to be calibrated
    desired_z = 0.06 # in meters, desired distance from the flower

    # gain for the pursuit controller
    kp_pos = 1.0 # need to be tuned, from pixel error to m/s
    kp_ang = 1.0 # need to be tuned, from propotional error to deg/s

    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(base_servo_mode)
        command = Base_pb2.TwistCommand()
        # note that twist is naturally in tool frame, but this conversion made things a bit easy
        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        command.duration = 0

        # initialize the twist command
        twist = command.twist
        twist.linear_x = 0.0  # adjust linear velocity
        twist.linear_y = 0.0
        twist.linear_z = 0.0
        twist.angular_x = 0.0
        twist.angular_y = 0.0
        twist.angular_z = 0.0  # adjust angular velocity

        # initialize the orientation of the end effector
        R_d = getRotMtx(base.GetMeasuredCartesianPose())

        R_ee_micro = euler_to_rotation_matrix(-np.radians(90), 0, np.radians(180))
        EE_polli_fork_tf = get_polli_fork_tf_from_yaml()
        H_ee_fork = tf_to_hom_mtx(EE_polli_fork_tf)

        cap = cv2.VideoCapture(4,cv2.CAP_V4L2)
        width = 1920
        height = 1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        REACHED = False

        while True:
            ret, frame = cap.read()
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # fit the flower center with an ellipse
            center, radius_list, frame = get_flower_center(frame, debug=False, show=True, REAL_FLOWER=REAL_FLOWER)
            
            H_wd_ee = get_world_EE_HomoMtx(base)
            R_wd_micro = H_wd_ee[:3,:3] @ R_ee_micro

            pix_error_micro, ang_error = calc_xy_plane_pose_error(center, radius_list)
            
            # print("Pix Error: ", pix_error_micro, "Ang Error: ", np.degrees(ang_error))

            flower_centered = np.linalg.norm(pix_error_micro[0:2]) < 250

            ang_error_for_Rot = 0 # no rotation for now
            # case: if the flower has rotated to almost upright
            if ang_error < np.radians(15):
                # ang_error_for_Rot = ang_error # correcting small angle error
                ang_error = 0
                # print("Small Angle Error: ", np.degrees(ang_error_for_Rot))
                if flower_centered:
                    REACHED = True
            
            # case: if the flower is not too far away from center
            if flower_centered: 
                pix_error_micro = np.zeros(3)
                # convert angle error to the micro +z error: map (0~90)*kp_ang to pixel error
                pix_error_micro[2] = -np.degrees(ang_error) * kp_ang

            print("area: ", radius_list[0]*radius_list[1])
            # case: if the flower center has not been detected
            if radius_list[0]*radius_list[1] < 10000:
                pix_error_micro = np.zeros(3)
                pix_error_micro[2] = -10 # move the fork up slowly
            

            pix_error_wd = R_wd_micro @ pix_error_micro * kp_pos
            pos_error_wd = pix_error_wd / focal_length * desired_z
            print("Position Error: ", pos_error_wd, "Angle Error: ", np.degrees(ang_error)*kp_ang)

            p_ee_fork = H_ee_fork[:3,3]

            # note that we are rotating around the fork center around the -x axis of the 
            # end effector frame
            H_wd_fork = H_wd_ee @ H_ee_fork
            H_error_fork_fork_des = rotate_frame_on_ball(p_ee_fork, -ang_error_for_Rot, 0, 0, centering=False)
            # print(H_error_fork_fork_des)
            H_error_wd_fork_des = H_wd_fork @ H_error_fork_fork_des

            # composing error rotation matrix from ang_error
            R_d_ee = (H_error_wd_fork_des @ np.linalg.inv(H_ee_fork))[:3,:3]
            ER_wd = H_wd_ee[:3,:3] @ R_d_ee.T

            # pos error in ee frame from rotating the fork
            pos_rot_error_wd = H_error_wd_fork_des[:3,3] - H_wd_fork[:3,3]
            # combining the displacement from both the position and the rotation
            pos_diff_wd = pos_error_wd + pos_rot_error_wd
            # print(pos_diff_wd)
            stopping, v, w = move_end_effector_vel(base, pos_diff_wd, ER_wd,
                                                    max_vel=0.008, max_w=0.5,
                                                    kp_pos=2.5, kp_ang=2.0,
                                                    eps_pos=0.001, eps_ang=0.1,
                                                    dcc_factor=2, ang_dcc_factor=1)
            # print("Velocity: ", v)
            # if REACHED:
            #     break

            twist.linear_x = v[0]
            twist.linear_y = v[1]
            twist.linear_z = v[2]
            twist.angular_x = w[0]
            twist.angular_y = w[1]
            twist.angular_z = w[2]
            base.SendTwistCommand(command)
            # center_list.append(center)
            # ellipse_rad_list.append(radius_list)
            # if len(center_list) > list_length:
                # center_list.pop(0)
                # ellipse_rad_list.pop(0)

            frame_rs = cv2.resize(frame.copy(), (540,960))
            cv2.imshow('frame', frame_rs)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()


def opt_ee_y_tilt(flower_point, all_flower_points, H_wd_rs, max_tilt=np.pi/6):
    ## using all flower points to calculate the mean
    # tilt of the ee should be the angle between the line connecting the flower point and the mean of all flower points in realsense frame
    H_rs_wd = np.linalg.inv(H_wd_rs)
    all_flower_points_rs = np.array([H_rs_wd[:3,:3] @ p + H_rs_wd[:3,3] for p in all_flower_points])
    flower_point_rs = H_rs_wd[:3,:3] @ flower_point + H_rs_wd[:3,3]
    
    mean_flower_point_rs = np.mean(all_flower_points_rs, axis=0)
    flower_vec_2d = flower_point_rs[:2] - mean_flower_point_rs[:2]
    # mirror about the x-axis, bring all positive y to negative
    angle_from_neg_y_axis = -np.pi /2 - np.arctan2(-np.abs(flower_vec_2d[1]), flower_vec_2d[0])
    angle_from_neg_y_axis = np.clip(angle_from_neg_y_axis, -max_tilt, max_tilt)
    # add minus sign so it's rotation about the positive z-axis of the camera frame
    return -angle_from_neg_y_axis


def robot_pollinate_single_flower(rs_flower_loc=None, _lambda = 0.5, serial_obj=None, euler_x=0):
    # rs_flower_loc is the location of the flower in the world frame
    # if it is None, the robot will move to the nearest flower in endoscope frame
    EE_endo_tf = get_endoscope_tf_from_yaml()
    EE_polli_fork_tf = get_polli_fork_tf_from_yaml()

    if rs_flower_loc is not None:
        # all pose below are in world frame
        H_wd_rs = get_current_RS_pose()
        p_rs = H_wd_rs[:3,3]  # realsense cam pose in world frame
        # draw a straight line from the realsense cam to the flower
        # put endoscope z-axis right on top of the line
        # put endoscope origin at _lambda:[0,1] along the line
        des_z = rs_flower_loc - p_rs
        # only consider x and y, setting z to the same as the flower
        des_z[2] = 0
        p_endo_des = rs_flower_loc - (1.0-_lambda) * des_z
        euler_z = np.arctan2(des_z[1], des_z[0])
        # euler_y = -np.arctan2(des_z[2], np.linalg.norm(des_z[:2]))
        euler_y = np.radians(-5)
        R_des_wd = euler_to_rotation_matrix(euler_x, euler_y, euler_z) # order is ZYX body frame rotation
        
        # trusting that this conversion between world and home camera conversion is correct
        R_wd_cam = euler_to_rotation_matrix(-np.radians(90), 0, -np.radians(90))
        
        H_endo_des_wd = np.eye(4)
        H_endo_des_wd[:3,:3] = R_des_wd @ R_wd_cam
        H_endo_des_wd[:3,3] = p_endo_des
        
        H_wd_ee_des, p_rs_des_kinova = robot_move_in_endoscope_frame_absolute(H_endo_des_wd)


    REAL_FLOWER = False
    if REAL_FLOWER:
        per_H = 0.5
    else:
        per_H = 0.75

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
    
    ### Visulization
    # draw_registration_result(template_pcd, flower_pcd, H_flower_in_endo)

    H_endo_des = rotate_frame_on_ball(H_flower_in_endo[:3,3], 0, flower_pitch, 0)
    print("Desired Endoscope Pose: \n", H_endo_des)

    H_wd_ee_des, p_orient_kinova = robot_move_in_endoscope_frame_relative(H_endo_des)
    # print("Desired Reorienting Pose: \n", p_orient_kinova)
    H_wd_ee_curr, p_curr_kinova = get_current_EE_pose()
    # print("Reorienting Pose error: ", p_orient_kinova - p_curr_kinova)


    # # draw_registration_result(template_pcd, flower_pcd, H_flower_in_endo)

    ################# Get to the bottom of the flower slowly #################
    ## Get the bottom of the flower pose in the fork frame after reorienting

    if REAL_FLOWER:
        fork_depth = 0.005  # the depth of the fork in the flower (0.007 mm) z is pointing out
        fork_lower = 0.001 # the raise of the fork from the lowest point of the flower (0.001 mm) y is pointing down
    else:
        fork_depth = 0.003  # the depth of the fork in the flower (0.007 mm) z is pointing out
        fork_lower = 0.001 # the raise of the fork from the lowest point of the flower (0.001 mm) y is pointing down

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

    # if bottom_z > p_flower_origin_fork_frame[2]: # if the flower is facing down

    H_polli_fork_des = np.eye(4)
    H_polli_fork_des[0:3, 3] = [p_flower_origin_fork_frame[0], bottom_y+fork_lower, extend_z+fork_depth]
    H_wd_polli_fork_des = H_wd_polli_fork_init @ H_polli_fork_des
    H_wd_ee_des, p_polli_fork_des_kinova = robot_move_in_polli_fork_frame_absolute(H_wd_polli_fork_des, speed=0.03)
    print("Desired Polli Fork Pose: \n", p_polli_fork_des_kinova)
    H_wd_ee_curr, p_curr_kinova = get_current_EE_pose()
    print("Move to bottom Pose error: ", p_polli_fork_des_kinova - p_curr_kinova)

    # micro adjust the robot to the flower
    robot_micro_adjust(serial_obj, REAL_FLOWER=REAL_FLOWER)


    ################# Keep moving slowly based on the vibration #################
    # try:
    #     cv2_video_display()
    # except KeyboardInterrupt:
    #     pass
    auto_focus(serial_obj,predefined_pos=200, predefined_zoom=40, real_flower=False)
    # input("Press Enter to continue...")

    ################# Return to the starting pose #################
    p_kinova_series = [p_polli_fork_des_kinova, p_orient_kinova, p_init_kinova]
    velocity_series = [0.03, None, None]  # None means default speed


    robot_move_kinova_pose_series(p_kinova_series, velocity_series)


def main():
    SerialObj = arduino_connect()
    # robot_micro_adjust(SerialObj, REAL_FLOWER=False)
    ##first raise robot's third joint to take pictures

    flower_poses_wd, joint_angles_init, H_wd_rs = realsense_get_flower_poses(sahi_n_slices = 2)
    print(flower_poses_wd)
    if flower_poses_wd is None:
        return
    for i in range(flower_poses_wd.shape[0]):
        # find the tilt angle of the microscope to get out of the way of plants
        tilt_angle = opt_ee_y_tilt(flower_poses_wd[i], flower_poses_wd, H_wd_rs)
        print("Tilt Angle: ", tilt_angle*180/np.pi)
        robot_pollinate_single_flower(rs_flower_loc=flower_poses_wd[i], 
                                      _lambda = 0.8,
                                      serial_obj=SerialObj,
                                      euler_x=tilt_angle)
        
    # move back to the initial pose
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)
        move_joints(base, joint_angles_init)
        

if __name__ == '__main__':
    main()