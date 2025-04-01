import cv2, threading
import numpy as np

from camera_path_6d import generate_upper_hemisphere_path_with_orientation
from cam_pose import get_world_cam_HomoMtx, capture_image, get_world_EE_HomoMtx, start_background_pose_capture, capture_pose_on_keypress

from gen3_7dof.tool_box import H_mtx_to_kinova_pose_in_base, move_tool_pose_absolute, move_tool_pose_relative, TCPArguments
from gen3_7dof.utilities import DeviceConnection
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from proj_farmhand.ICP_tool_box import rotate_frame_on_ball


"""
There are two ways to move the robot in this script: Path-following and Teleoperation
For Path-following, right now we only allow rotations about the global Y-axis
"""


# Mode 1: Path-following
def generate_pose_matrices(center, radius, num_points):
    path_points = generate_upper_hemisphere_path_with_orientation(radius, num_points)
    pose_matrices = []

    for _, point in enumerate(path_points):
        x, y, z, yaw = point
        position = center + np.array([x, y, z])

        delta = position - center  # Direction vector from the flower center to camera
        pitch_angle = np.arctan2(delta[0], delta[2])
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


# Mode 2: Teleoperation
def teleop_camera(center=np.array([0, 0, 10]), pos_step=0.01, speed=0.03):

    print("Starting teleop... ")

    """
    Allow manual control of the robot using keyboard:
    - 'w': move forward (+Y)
    - 's': move backward (-Y)
    - 'e': move forward (+X)
    - 'q': move backward (-X)
    - 'a': move forward (+Z)
    - 'd': move backward (-Z)
    - 'c': capture an image
    - 'x': exit teleop

    To rotate about 10 cm +z in camera frame, need to provide ball center
    reference: rotate_frame_on_ball(H_flower_in_endo[:3,3], 0, flower_pitch, 0)
    => H_flower_in_endo is the transformation matrix from flower to endoscope, so [0, 0, 10]
    """

    # TODO: Have problem turning on the camera
    # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # Establish connection to the Kinova robot
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        # Create a dummy black window (can be used for visual feedback later)
        cv2.namedWindow("Teleop", cv2.WINDOW_NORMAL)
        black = np.zeros((200, 400, 3), dtype=np.uint8)

        while True:
            # ret, frame = cap.read()
            # if ret:
            #     cv2.imshow("Teleop", frame)

            cv2.imshow("Teleop", black)

            key = cv2.waitKey(10) & 0xFF

            motion = None

            if key == ord('x'):
                print('Exiting...')
                break
            
            # TODO: Need to fix the camera capture
            elif key == ord('c'):
                H_world_EE = get_world_EE_HomoMtx(base)
                camera_pose = get_world_cam_HomoMtx(H_world_EE)
                capture_image(camera_pose)
            
            elif key == ord('w'):
                print("Moving +Y")
                motion = [0, pos_step, 0, 0, 0, 0]  # 6-element array: [x, y, z, roll, pitch, yaw]

            elif key == ord('s'):
                print("Moving -Y")
                motion = [0, -pos_step, 0, 0, 0, 0]

            elif key == ord('q'):
                print("Moving -X")
                motion = [-pos_step, 0, 0, 0, 0, 0]

            elif key == ord('e'):
                print("Moving +X")
                motion = [pos_step, 0, 0, 0, 0, 0]

            elif key == ord('a'):
                print("Moving +Z")
                motion = [0, 0, pos_step, 0, 0, 0]

            elif key == ord('d'):
                print("Moving -Z")
                motion = [0, 0, -pos_step, 0, 0, 0]

            if motion:
                move_tool_pose_relative(base, base_cyclic, motion, speed)

        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    center = np.array([1.0, 2.0, 0.0])
    radius = 0.12
    num_points = 10
    mode = "teleop"

    if mode == "path_following":
        move_camera_on_path(center=center, radius=radius, num_points=num_points)
    elif mode == "teleop":
        teleop_camera(center=center, pos_step=0.01, speed=0.03)
        