import cv2
import threading
import math
import numpy as np

from camera_path_6d import generate_upper_hemisphere_path_with_orientation
from cam_pose import get_world_cam_HomoMtx, get_world_EE_HomoMtx, capture_image, start_background_pose_capture

from geometry_msgs.msg import TransformStamped
from gen3_7dof.tool_box import rotation_matrix_to_euler, H_mtx_to_kinova_pose_in_base, tf_to_hom_mtx, move_tool_pose_absolute, move_tool_pose_relative, TCPArguments
from gen3_7dof.utilities import DeviceConnection
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from proj_farmhand.ICP_tool_box import rotate_frame_on_ball


class MoveRobot:
    def __init__(self, center=np.array([0.0, 0.0, 10.0]), radius=0.12, device_id=2):
        self.center = center
        self.radius = radius
        self.device_id = device_id

        # Initialize camera
        # device_id = 2 if using laptop
        self.cap = cv2.VideoCapture(self.device_id)

        # Initialize Kinova robot connection
        tcp_args = TCPArguments()
        self.router = DeviceConnection.createTcpConnection(tcp_args)
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)

        # Initialize orientation
        direction = self.center / np.linalg.norm(self.center)
        self.pitch = np.arctan2(direction[0], direction[2])
        self.yaw = np.arctan2(direction[1], direction[0])
        self.roll = 0

        self.EE_endo_tf = self.get_endoscope_tf()
        self.pose_log = []


    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.router.close()
 

    """
    Mode 1: Path-Following
    # Right now we only allow rotations about the global Y-axis!
    """
    def move_camera_on_path(self, num_points=10, speed=0.05, capture=True):
        print("Starting path following...")
        poses = generate_upper_hemisphere_path_with_orientation(self.radius, num_points)

        if capture:
            stop_event = threading.Event()
            capture_thread = threading.Thread(target=start_background_pose_capture, args=(1.0, stop_event))
            capture_thread.start()

        try:
            for point in poses:
                x, y, z, yaw = point
                position = self.center + np.array([x, y, z])
                direc = position - self.center
                pitch_angle = np.arctan2(direc[0], direc[2])
                roll_angle = 0

                H = rotate_frame_on_ball(self.center, roll=roll_angle, pitch=pitch_angle, yaw=yaw)
                kinova_pose = H_mtx_to_kinova_pose_in_base(H)
                move_tool_pose_absolute(self.base, kinova_pose, speed)
        finally:
            if capture:
                stop_event.set()
                capture_thread.join()


    """
    Mode 2: Free-space Teleoperation 
    """
    def free_space_teleop(self, pos_step=0.01, rot_step=1, speed=0.03):
        print("Starting free-space teleop...")
        while True:
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow("Teleop", frame)

            key = cv2.waitKey(10) & 0xFF
            motion = None

            if key == ord('x'):
                print("Exiting...")
                break

            elif key == ord('c'):
                self.capture_image()

            # Transition
            elif key in [ord('w'), ord('s'), ord('e'), ord('q'), ord('a'), ord('d')]:
                # 6-element array: [x, y, z, roll, pitch, yaw]
                motion = [0, 0, 0, 0, 0, 0]
                if key == ord('w'):
                    motion[1] = pos_step
                elif key == ord('s'):
                    motion[1] = -pos_step
                elif key == ord('e'):
                    motion[0] = pos_step
                elif key == ord('q'):
                    motion[0] = -pos_step
                elif key == ord('a'):
                    motion[2] = pos_step
                elif key == ord('d'):
                    motion[2] = -pos_step

            # Rotation
            elif key in [ord('u'), ord('j'), ord('i'), ord('k'), ord('o'), ord('l')]:
                motion = [0, 0, 0, 0, 0, 0]
                if key == ord('u'):
                    motion[3] = rot_step
                elif key == ord('j'):
                    motion[3] = -rot_step
                elif key == ord('i'):
                    motion[4] = rot_step
                elif key == ord('k'):
                    motion[4] = -rot_step
                elif key == ord('o'):
                    motion[5] = rot_step
                elif key == ord('l'):
                    motion[5] = -rot_step

            if motion:
                move_tool_pose_relative(self.base, self.base_cyclic, motion, speed)


    """
    Mode 3: Teleoperation on Sphere
    """
    def get_endoscope_tf(self):
        EE_endo_tf = TransformStamped()
        EE_endo_tf.header.frame_id = "end_effector"
        EE_endo_tf.child_frame_id = "endoscope"
        EE_endo_tf.transform.translation.x = 0.0
        EE_endo_tf.transform.translation.y = -0.05
        EE_endo_tf.transform.translation.z = 0.11
        EE_endo_tf.transform.rotation.x = 0.0
        EE_endo_tf.transform.rotation.y = 0.0
        EE_endo_tf.transform.rotation.z = 1.0
        EE_endo_tf.transform.rotation.w = 0.0

        return EE_endo_tf


    def small_rotation(self, axis, angle_rad):
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)

        if axis == 'x':
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 'y':
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:
            raise ValueError("Invalid axis: choose 'x' or 'y'")

        return R
    

    def save_pose_log(self, filename="camera_poses.npy"):
        """
        Save all logged camera poses to a .npy file.
        """
        poses_array = np.array(self.pose_log)  # shape (N, 4, 4)
        np.save(filename, poses_array)
        print(f"[INFO] Saved {len(self.pose_log)} poses to {filename}.")
    

    def teleop_on_sphere(self, pitch_step=0.5, yaw_step=0.5, speed=0.03):
        print("Starting teleop on sphere...")

        while True:
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow("Sphere Teleop", frame)

            key = cv2.waitKey(10) & 0xFF

            H_wd_ee = get_world_EE_HomoMtx(self.base)
            H_wd_cam = H_wd_ee @ tf_to_hom_mtx(self.EE_endo_tf)
            H_cam_delta = np.eye(4)

            if key == ord('x'):
                self.save_pose_log("teleop_camera_poses.npy")
                print("Exiting sphere teleop...")
                break
            elif key == ord('c'):
                self.capture_image()

            elif key == ord('i'):
                # self.pitch += math.radians(pitch_step)
                H_cam_delta[:3, :3] = self.small_rotation('x', math.radians(pitch_step))
            elif key == ord('k'):
                H_cam_delta[:3, :3] = self.small_rotation('x', -math.radians(pitch_step))
            elif key == ord('j'):
                H_cam_delta[:3, :3] = self.small_rotation('y', math.radians(yaw_step))
            elif key == ord('l'):
                H_cam_delta[:3, :3] = self.small_rotation('y', -math.radians(yaw_step))

            if not np.allclose(H_cam_delta, np.eye(4)):
                H_wd_cam_des = H_wd_cam @ H_cam_delta
                self.robot_move_in_camera_frame_relative(H_wd_cam_des, speed=speed)
                self.pose_log.append(H_wd_cam_des.copy())


    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            H_world_EE = get_world_EE_HomoMtx(self.base)  # Get the EE's homogeneous matrix in world frame
            cam_pose = get_world_cam_HomoMtx(H_world_EE)  # Get the camera's homogeneous matrix in world frame
            capture_image(cam_pose, ret=ret, frame=frame)


    def robot_move_in_camera_frame_relative(self, H_cam_desired, speed=None):
        """
        Returns the default hardcoded transform from end-effector to camera
        The camera is:
        - Rotated 180° about the Z-axis of the EE frame
        - Translated -0.05m along EE Y and +0.11m along EE Z
        """
        H_wd_cam_des = H_cam_desired
        H_wd_ee_des = H_wd_cam_des @ np.linalg.inv(tf_to_hom_mtx(self.EE_endo_tf))

        p_world = H_wd_ee_des[:3, 3]
        R_ee = H_wd_ee_des[:3, :3]
        r_wd, p_wd, y_wd = rotation_matrix_to_euler(R_ee)
        r_wd, p_wd, y_wd = np.degrees(r_wd), np.degrees(p_wd), np.degrees(y_wd)

        p_des_kinova = np.array([p_world[0], p_world[1], p_world[2], r_wd, p_wd, y_wd])
        move_tool_pose_absolute(self.base, p_des_kinova, speed=speed)


if __name__ == "__main__":
    robot = MoveRobot(center=np.array([0.0, 0.0, 10.0]))
    robot.teleop_on_sphere()
