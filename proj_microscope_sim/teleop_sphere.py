import os
import cv2
import math
import numpy as np
from datetime import datetime

from cam_pose import get_world_cam_HomoMtx, get_world_EE_HomoMtx

from geometry_msgs.msg import TransformStamped
from gen3_7dof.tool_box import rotation_matrix_to_euler, tf_to_hom_mtx, move_tool_pose_absolute, TCPArguments
from gen3_7dof.utilities import DeviceConnection
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2


class MoveRobot:
    def __init__(self, center, save_dir, radius=0.12, device_id=2):
        self.center = center
        self.save_dir = save_dir
        self.radius = radius
        self.device_id = device_id

        # Generate timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"session_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)


        # Initialize camera
        # device_id = 2 if using laptop
        try:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError(f"[ERROR] Failed to open camera device {self.device_id}")
            
            print(f"[INFO] Camera {self.device_id} opened successfully.")
        
        except Exception as e:
            print(f"[WARN] Camera initialization failed: {e}")
            self.cap = None

        
        # Initialize a cv2.VideoWriter object
        self.video_writer = None
        self.video_recording = False


        # Initialize orientation
        direction = self.center / np.linalg.norm(self.center)
        self.pitch = np.arctan2(direction[0], direction[2])
        self.yaw = np.arctan2(direction[1], direction[0])
        self.roll = 0.0

        self.EE_endo_tf = self.get_EE_endoscope_tf()
        self.image_counter = 0
        self.pose_log = []
        self.captured_poses = []


    def __del__(self):
        try:
            if self.cap:
                self.cap.release()
                print("[INFO] Camera released.")
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
            print("[INFO] OpenCV windows closed.")
        except Exception:
            pass

        try:
            if self.video_writer:
                self.video_writer.release()
                print("[INFO] Video writer released.")
        except Exception:
            pass

    
    def get_EE_endoscope_tf(self):
        """
        Returns the default hardcoded transform from end-effector to camera
        The camera is:
        - Rotated 180° about the Z-axis of the EE frame
        - Translated -0.05m along EE Y and +0.11m along EE Z
        """
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
        elif axis == 'z':
            R = np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])
        else:
            raise ValueError("Invalid axis: choose 'x', 'y' or 'z'")

        return R
    

    def save_pose_log(self, filename="poses_bounds.npy", near=0.05, far=0.20):
        """
        Save all logged camera poses to LLFF-compatible poses_bounds.npy.
        Each pose becomes a 3x5 matrix: [right | up | back | position]
        """
        poses_bounds = []
        for H in self.captured_poses:
            R = H[:3, :3]
            t = H[:3, 3]
            right = R[:, 0]
            up = R[:, 1]
            back = -R[:, 2]  # NeRF expects camera looking down -Z
            pose_3x5 = np.stack([right, up, back, t], axis=1)
            pose_flat = pose_3x5.flatten()
            pose_bounds = np.concatenate([pose_flat, [near, far]])
            poses_bounds.append(pose_bounds)

        poses_bounds = np.array(poses_bounds)

        save_path = os.path.join(self.save_dir, filename)
        np.save(save_path, poses_bounds)
        print(f"[INFO] Saved {len(poses_bounds)} poses to {filename}.")

        assert poses_bounds.shape[0] == self.image_counter, \
            f"[ERROR] Expected {self.image_counter} poses, but got {poses_bounds.shape[0]}"
        assert poses_bounds.shape[1] == 14, \
            f"[ERROR] Each pose should have 14 values (3x4 + near + far), got {poses_bounds.shape[1]}"

        print(f"[INFO] Verified {poses_bounds.shape[0]} poses written to {save_path}")


    def teleop_on_sphere(self, pitch_step=0.5, yaw_step=0.5, speed=0.03):
        print("[INFO] Starting teleop on sphere…")

        tcp_args = TCPArguments()
        with DeviceConnection.createTcpConnection(tcp_args) as router:
            base = BaseClient(router)
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            base.SetServoingMode(base_servo_mode)

            while True:
                ret, frame = self.cap.read()
                if ret:
                    cv2.imshow("Sphere Teleop", frame)

                key = cv2.waitKey(10) & 0xFF

                H_wd_ee = get_world_EE_HomoMtx(base)
                init_H_wd_cam = H_wd_ee @ tf_to_hom_mtx(self.EE_endo_tf)
                H_cam_flower = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0.1],  # +10 cm in Z
                                        [0, 0, 0, 1]
                                        ])
                H_wd_flower = init_H_wd_cam @ H_cam_flower
                H_cam_delta = np.eye(4)

                moved = False
               
                if key == ord('x'):
                    self.save_pose_log("camera_poses.npy")
                    print("[INFO] Exiting sphere teleop.")
                    break

                elif key == ord('c'):
                    self.capture_image()

                elif key == ord('v'):
                    self.start_video_recording()

                    """
                    Note:
                    - Rotation axes ('x', 'y', 'z') are defined in the world frame
                    - The delta angle is applied in the camera frame
                    """

                elif key == ord('k'):
                    H_cam_delta[:3, :3] = self.small_rotation('z', math.radians(pitch_step))
                    moved = True

                elif key == ord('i'):
                    H_cam_delta[:3, :3] = self.small_rotation('z', -math.radians(pitch_step))
                    moved = True
                    
                elif key == ord('l'):
                    H_cam_delta[:3, :3] = self.small_rotation('y', math.radians(yaw_step))
                    moved = True
                    
                elif key == ord('j'):
                    H_cam_delta[:3, :3] = self.small_rotation('y', -math.radians(yaw_step))
                    moved = True

                if moved:
                    H_wd_cam = self.pose_log[-1] if self.pose_log else init_H_wd_cam
                    
                    # Rotate the camera pose around the flower center
                    T_to_center = np.eye(4)
                    T_to_center[:3, 3] = -H_wd_flower[:3, 3]

                    T_from_center = np.eye(4)
                    T_from_center[:3, 3] = H_wd_flower[:3, 3]

                    H_wd_cam_des = T_from_center @ H_cam_delta @ T_to_center @ H_wd_cam

                    # Move robot once and log pose
                    self.robot_move_to_camera_pose(base, H_wd_cam_des, speed=speed)
                    self.pose_log.append(H_wd_cam_des.copy())


    def capture_image(self):
        ret, frame = self.cap.read()

        if len(self.pose_log) == 0:
            print("[WARN] No pose available to save.")
            return
        
        if ret:
            # Save image
            img_filename = os.path.join(self.save_dir, f"img_{self.image_counter:04d}.png")
            cv2.imwrite(img_filename, frame)

            # Save pose
            self.captured_poses.append(self.pose_log[-1].copy())
            print(f"[INFO] Saved img_{self.image_counter:04d}.png and pose")
            
            self.image_counter += 1


    def start_video_recording(self, filename="output.avi", fps=10):
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MJPG' or 'mp4v'
        save_path = os.path.join(self.save_dir, filename)
        self.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        self.video_recording = True

        if not self.video_writer.isOpened():
            raise RuntimeError(f"[ERROR] Failed to open video file: {save_path}")
        
        print(f"[INFO] Video recording started: {save_path}")


    def robot_move_to_camera_pose(self, base, H_wd_cam_des, speed=None):
        H_wd_ee_des = H_wd_cam_des @ np.linalg.inv(tf_to_hom_mtx(self.EE_endo_tf))

        p_world = H_wd_ee_des[:3, 3]
        R_ee = H_wd_ee_des[:3, :3]
        r_wd, p_wd, y_wd = rotation_matrix_to_euler(R_ee)
        r_wd, p_wd, y_wd = np.degrees(r_wd), np.degrees(p_wd), np.degrees(y_wd)

        p_des_kinova = np.array([p_world[0], p_world[1], p_world[2], r_wd, p_wd, y_wd])
        move_tool_pose_absolute(base, p_des_kinova, speed=speed)


if __name__ == "__main__":
    robot = MoveRobot(center=np.array([0.0, 0.0, 0.1]), save_dir='data')
    robot.teleop_on_sphere()