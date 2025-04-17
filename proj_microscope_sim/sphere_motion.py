import cv2, threading, math
import numpy as np

from camera_path_6d import generate_upper_hemisphere_path_with_orientation
from cam_pose import get_world_cam_HomoMtx, get_world_EE_HomoMtx, get_EE_cam_HomoMtx, capture_image, start_background_pose_capture

from geometry_msgs.msg import TransformStamped

from gen3_7dof.tool_box import rotation_matrix_to_euler
from gen3_7dof.tool_box import H_mtx_to_kinova_pose_in_base, tf_to_hom_mtx
from gen3_7dof.tool_box import move_tool_pose_absolute, move_tool_pose_relative, TCPArguments
from gen3_7dof.utilities import DeviceConnection
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from proj_farmhand.ICP_tool_box import rotate_frame_on_ball


"""
For Path-following, right now we only allow rotations about the global Y-axis
"""


# Mode 1: Path-following
def generate_pose_matrices(center, radius, num_points):
    path_points = generate_upper_hemisphere_path_with_orientation(radius, num_points)
    pose_matrices = []

    for _, point in enumerate(path_points):
        x, y, z, yaw = point
        position = center + np.array([x, y, z])

        direction = position - center  # Direction vector from the flower center to camera
        pitch_angle = np.arctan2(direction[0], direction[2])
        roll_angle = 0
        yaw_angle = yaw

        # Get a transformation matrix H that orients the camera to face the flower (center)
        H = rotate_frame_on_ball(center, roll=roll_angle, pitch=pitch_angle, yaw=yaw_angle)
        
        pose_matrices.append(H)

    return pose_matrices


def move_camera_on_path(center, radius=0.12, num_points=10, speed=0.05, capture=True):
    # Establish connection to the Kinova robot
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)

    # Make sure the robot is in servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    poses = generate_pose_matrices(center=center, radius=radius, num_points=num_points)

    # Start camera capture in background thread
    if capture:
        stop_event = threading.Event()
        capture_thread = threading.Thread(target=start_background_pose_capture, args=(1.0, stop_event))
        capture_thread.start()

    try:
        for i, H in enumerate(poses):
            # Convert each pose to Kinova format
            kinova_pose = H_mtx_to_kinova_pose_in_base(H)
            print(f"[{i+1}/{len(poses)}] Moving to: {np.round(kinova_pose, 3)}")
            move_tool_pose_absolute(base, kinova_pose, speed=speed)


    finally:
        if capture:
            stop_event.set()
            capture_thread.join()
            print("Capture thread stopped.")


# Mode 2: Free-space Teleoperation 
def free_space_teleop(pos_step=0.01, rot_step=1, speed=0.03):
    try:
        print("Starting free-space teleop... ")

        """
        Function: Teleop the camera in free space

        Keys:
        - w/s: move +Y/-Y
        - e/q: move +X/-X
        - a/d: move +Z/-Z
        - u/j: rotate +X/-X
        - i/k: rotate +Y/-Y
        - o/l: rotate +Z/-Z
        - c: capture an image
        - x: exit teleop
        """
        
        """
        To rotate about 10 cm +z in camera frame, need to provide ball center
        reference: rotate_frame_on_ball(H_flower_in_endo[:3,3], 0, flower_pitch, 0)
        => H_flower_in_endo is the transformation matrix from flower to endoscope, so [0, 0, 10]
        """

        # Initialize camera
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(2)
        # Establish connection to the Kinova robot
        tcp_args = TCPArguments()
        with DeviceConnection.createTcpConnection(tcp_args) as router:
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router)

            while True:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow("Teleop", frame)

                key = cv2.waitKey(10) & 0xFF

                motion = None
                if key == ord('x'):
                    print('Exiting...')
                    break
                
                elif key == ord('c'):
                    H_world_EE = get_world_EE_HomoMtx(base)  # Get the EE's homogeneous matrix in world frame
                    cam_pose_in_world = get_world_cam_HomoMtx(H_world_EE)  # Get the camera's homogeneous matrix in world frame
                    capture_image(cam_pose_in_world, ret=ret, frame=frame)
                
                elif key == ord('w'):
                    print("Moving +Y")
                    motion = [0, pos_step, 0, 0, 0, 0]  # 6-element array: [x, y, z, roll, pitch, yaw]

                elif key == ord('s'):
                    print("Moving -Y")
                    motion = [0, -pos_step, 0, 0, 0, 0]

                elif key == ord('e'):
                    print("Moving +X")
                    motion = [pos_step, 0, 0, 0, 0, 0]

                elif key == ord('q'):
                    print("Moving -X")
                    motion = [-pos_step, 0, 0, 0, 0, 0]

                elif key == ord('a'):
                    print("Moving +Z")
                    motion = [0, 0, pos_step, 0, 0, 0]

                elif key == ord('d'):
                    print("Moving -Z")
                    motion = [0, 0, -pos_step, 0, 0, 0]

                elif key == ord('u'):
                    motion = [0, 0, 0, rot_step, 0, 0]

                elif key == ord('j'):
                    motion = [0, 0, 0, -rot_step, 0, 0]

                elif key == ord('i'):
                    motion = [0, 0, 0, 0, rot_step, 0]

                elif key == ord('k'):
                    motion = [0, 0, 0, 0, -rot_step, 0]

                elif key == ord('o'):
                    motion = [0, 0, 0, 0, 0, rot_step]
                    
                elif key == ord('l'):
                    motion = [0, 0, 0, 0, 0, -rot_step]

                if motion:
                    move_tool_pose_relative(base, base_cyclic, motion, speed)


    except KeyboardInterrupt:
        print("Process interrupted by user.")


    finally:
        cap.release()
        cv2.destroyAllWindows()


# Mode 3: Teleop on sphere
def get_endoscope_tf():
    """
    Returns the default hardcoded transform from end-effector to camera
    The camera is:
    - Rotated 180° about the Z-axis of the EE frame
    - Translated -0.05m along EE Y and +0.11m along EE Z
    """

    tf = TransformStamped()
    tf.header.frame_id = "end_effector"
    tf.child_frame_id = "endoscope"
    
    # Translation (meters)
    tf.transform.translation.x = 0.0
    tf.transform.translation.y = -0.05
    tf.transform.translation.z = 0.11

    # Rotation: 180° about Z → quaternion (0, 0, 1, 0)
    tf.transform.rotation.x = 0.0
    tf.transform.rotation.y = 0.0
    tf.transform.rotation.z = 1.0
    tf.transform.rotation.w = 0.0

    return tf

# TODO:
# H_cam_des is the delta, not abs position
# write this into a class
# Rotation is still weird (roll, pitch, yaw?)

def robot_move_in_camera_frame_relative(base, H_cam_des, speed=None):
    # note that the H_cam_des mapes between current camera pose and desired camera pose
    # so the input is NOT in world frame
    EE_endo_tf = get_endoscope_tf()
        
    # Make sure the arm is in Single Level Servoing mode (high-level mode)
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    H_wd_ee = get_world_EE_HomoMtx(base)
    H_wd_cam = H_wd_ee @ tf_to_hom_mtx(EE_endo_tf) # get the camera pose in world frame
    # print("Endoscope Pose in World Frame: \n", H_wd_endo)

    H_wd_cam_des = H_wd_cam @ H_cam_des
    # print("Desired Endoscope Pose in World Frame: \n", H_wd_endo_des)

    H_wd_ee_des = H_wd_cam_des @ np.linalg.inv(tf_to_hom_mtx(EE_endo_tf))
    p_world = H_wd_ee_des[:3,3]
    R_ee = H_wd_ee_des[:3,:3]

    r_wd, p_wd, y_wd = rotation_matrix_to_euler(R_ee)
    r_wd, p_wd, y_wd = np.degrees(r_wd), np.degrees(p_wd), np.degrees(y_wd)
    p_des_kinova = np.array([p_world[0], p_world[1], p_world[2], r_wd, p_wd, y_wd])
    print("Desired Reorienting Pose: \n", p_des_kinova)

    move_tool_pose_absolute(base, p_des_kinova, speed=speed)
    # return H_wd_ee_des, p_des_kinova


def teleop_on_sphere(center, radius=0.12, pitch_step=0.5, yaw_step=0.5, speed=0.03):
    try:
        # TODO: Still not rotate about the center of the sphere (no radius comes in)
        print("Starting teleop on sphere...")

        """
        Function:
        - Keyboard teleop along the sphere (fixed radius from center)
        - Only pitch and yaw are allowed to change

        Keys:
        - i/k: pitch up/down
        - j/l: yaw left/right
        - u/o: roll left/right
        - c: capture image
        - x: exit
        """

        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(2)  # on laptop

        tcp_args = TCPArguments()
        with DeviceConnection.createTcpConnection(tcp_args) as router:
            base = BaseClient(router)

            H_world_EE = get_world_EE_HomoMtx(base)  # Get the EE's homogeneous matrix in world frame
            camera_pose_in_world = get_world_cam_HomoMtx(H_world_EE)  # Get the camera's homogeneous matrix in world frame

            # cam_pose = np.array([
            #     camera_pose_in_world["camera_x"],
            #     camera_pose_in_world["camera_y"],
            #     camera_pose_in_world["camera_z"]
            # ])

            direction = center / np.linalg.norm(center)

            # Initial angles based on the direction vector in camera frame
            pitch = np.arctan2(direction[0], direction[2])
            roll = 0  # Rotation around viewing axis (z-axis in camera frame)
            yaw = np.arctan2(direction[1], direction[0])

            while True:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow("Sphere Teleop", frame)

                key = cv2.waitKey(10) & 0xFF

                pitch_motion = None
                yaw_motion = None
                if key == ord('x'):
                    print("Exiting...")
                    break

                elif key == ord('c'):
                    capture_image(camera_pose_in_world, ret=ret, frame=frame)

                elif key == ord('i'):
                    print("Pitching up")
                    pitch_motion = pitch + math.radians(pitch_step)
                    # pitch += math.radians(pitch_step)

                elif key == ord('k'):
                    pitch_motion = pitch - math.radians(pitch_step)
                    # pitch -= math.radians(pitch_step)

                elif key == ord('j'):
                    yaw_motion = yaw + math.radians(yaw_step)
                    # yaw += math.radians(yaw_step)

                elif key == ord('l'):
                    yaw_motion = yaw - math.radians(yaw_step)
                    # yaw -= math.radians(yaw_step)
                
                if pitch_motion or yaw_motion:
                    H_cam_desired = rotate_frame_on_ball(center, roll=roll, pitch=pitch_motion or pitch, yaw=yaw_motion or yaw)
                    robot_move_in_camera_frame_relative(base, H_cam_desired, speed=speed)

                    if pitch_motion:
                        pitch = pitch_motion
                    if yaw_motion:
                        yaw = yaw_motion


    except KeyboardInterrupt:
        print("Process interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    center = np.array([0.0, 0.0, 10.0])  # Center of the sphere/object in camera frame
    radius = 0.0001
    num_points = 10
    mode = "teleop_on_sphere"

    if mode == "path_following":
        # TODO: Need to check the usage of center
        move_camera_on_path(center=center, radius=radius, num_points=num_points)

    elif mode == "free_space_teleop":
        free_space_teleop(pos_step=0.01, rot_step=1, speed=0.03)

    elif mode == "teleop_on_sphere":
        teleop_on_sphere(center=center, radius=radius, pitch_step=0.5, yaw_step=0.5, speed=0.03)
        