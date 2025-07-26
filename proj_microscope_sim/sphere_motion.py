import cv2
import threading, os
import math
import numpy as np
from datetime import datetime

from camera_path_6d import generate_upper_hemisphere_path_with_orientation
from cam_pose import get_world_EE_HomoMtx, start_background_pose_capture

from geometry_msgs.msg import TransformStamped
from gen3_7dof.tool_box import rotation_matrix_to_euler, H_mtx_to_kinova_pose_in_base, tf_to_hom_mtx, move_tool_pose_absolute, move_tool_pose_relative, TCPArguments
from gen3_7dof.utilities import DeviceConnection
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from proj_farmhand.ICP_tool_box import rotate_frame_on_ball


class MoveRobot:
    def __init__(self, save_dir, device_id=3):
        self.device_id = device_id

        # Generate timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"session_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize camera; device_id = 2 if using laptop
        try:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError(f"[ERROR] Failed to open camera device {self.device_id}")
            print(f"[INFO] Camera {self.device_id} opened successfully")
        
        except Exception as e:
            print(f"[WARN] Camera initialization failed: {e}")
            self.cap = None

        # Initialize a cv2.VideoWriter object
        self.video_writer = None
        self.recording = False

        self.EE_cam_tf = self.get_EE_camera_tf()
        self.image_counter = 0
        self.pose_log = []
        self.captured_poses = []


    def __del__(self):
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                print("[INFO] Camera released")
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
            print("[INFO] OpenCV windows closed")
        except Exception:
            pass


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
            tcp_args = TCPArguments()
            with DeviceConnection.createTcpConnection(tcp_args) as router:
                base = BaseClient(router)
                base_servo_mode = Base_pb2.ServoingModeInformation()
                base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
                base.SetServoingMode(base_servo_mode)

                for point in poses:
                    x, y, z, yaw = point
                    position = self.center + np.array([x, y, z])
                    direc = position - self.center
                    pitch_angle = np.arctan2(direc[0], direc[2])
                    roll_angle = 0

                    H = rotate_frame_on_ball(self.center, roll=roll_angle, pitch=pitch_angle, yaw=yaw)
                    kinova_pose = H_mtx_to_kinova_pose_in_base(H)
                    move_tool_pose_absolute(base, kinova_pose, speed)
        
        finally:
            if capture:
                stop_event.set()
                capture_thread.join()


    """
    Mode 2: Free-space Teleoperation 
    """
    def free_space_teleop(self, pos_step=0.01, rot_step=1, speed=0.03):
        print("Starting free-space teleop...")
        tcp_args = TCPArguments()
        with DeviceConnection.createTcpConnection(tcp_args) as router:
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router)
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            base.SetServoingMode(base_servo_mode)

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
                    move_tool_pose_relative(base, base_cyclic, motion, speed)


    """
    Mode 3: Teleoperation on Sphere
    """
    def get_EE_camera_tf(self):
        """
        Returns the default hardcoded transform from end-effector to camera
        The camera is:
        - Rotated 180° about the Z-axis of the EE frame
        - Translated -0.05m along EE Y and +0.11m along EE Z
        """
        EE_cam_tf = TransformStamped()
        EE_cam_tf.header.frame_id = "end_effector"
        EE_cam_tf.child_frame_id = "camera"
        EE_cam_tf.transform.translation.x = 0.0
        EE_cam_tf.transform.translation.y = -0.05
        EE_cam_tf.transform.translation.z = 0.11
        EE_cam_tf.transform.rotation.x = 0.0
        EE_cam_tf.transform.rotation.y = 0.0
        EE_cam_tf.transform.rotation.z = 1.0
        EE_cam_tf.transform.rotation.w = 0.0
        
        return EE_cam_tf


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
            raise ValueError("Invalid axis")

        return R
    

    # RAFT_tool_box.py reports a focal length of 474.2788
    def save_pose_log(self, filename="poses_bounds.npy", image_size=(480, 640), focal=474.2788, near=0.05, far=0.20):
        """
        Save captured poses to LLFF-style poses_bounds.npy with:
        - 3x4 camera-to-world matrix (in NeRF format)
        - appended intrinsics: [H, W, focal]
        - appended near/far bounds
        => total 17 values per row
        """

        H, W = image_size
        poses_bounds = []

        for H_wd_cam in self.captured_poses:
            H_cam_wd = np.linalg.inv(H_wd_cam)
            R = H_cam_wd[:3, :3]
            t = H_cam_wd[:3, 3]

            # Reorder rotation columns to match LLFF's convention
            right = R[:, 0]
            down = R[:, 1]
            back = -R[:, 2]  # NeRF expects camera looking down -Z

            # Construct 3x5 matrix
            pose_intrinsics = np.stack([down, right, back, t, np.array([H, W, focal])], axis=1)  # (3, 5)
            pose_flat = pose_intrinsics.flatten()

            # Append near/far
            pose_bounds = np.concatenate([pose_flat, [near, far]])
            poses_bounds.append(pose_bounds)

        poses_bounds = np.stack(poses_bounds)
        save_path = os.path.join(self.save_dir, filename)
        np.save(save_path, poses_bounds)
    

    def teleop_on_sphere(self, pitch_step=0.5, yaw_step=0.5, speed=0.03):
        tcp_args = TCPArguments()
        with DeviceConnection.createTcpConnection(tcp_args) as router:
            base = BaseClient(router)
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            base.SetServoingMode(base_servo_mode)

            warmup_frames = 30
            for _ in range(warmup_frames):
                ret, frame = self.cap.read()

            while True:
                ret, frame = self.cap.read()
                if ret:
                    cv2.imshow("Sphere Teleop", frame)

                    if self.recording and self.video_writer is not None:
                        self.video_writer.write(frame)

                key = cv2.waitKey(10) & 0xFF

                H_wd_ee = get_world_EE_HomoMtx(base)
                init_H_wd_cam = H_wd_ee @ tf_to_hom_mtx(self.EE_cam_tf)

                moved = False
                H_cam_flower = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0.1],  # +10 cm in Z
                                        [0, 0, 0, 1]
                                        ])
                H_wd_flower = init_H_wd_cam @ H_cam_flower
                H_cam_delta = np.eye(4)
               
                if key == ord('x'):
                    self.save_pose_log("camera_poses.npy")
                    print("[INFO] Exiting sphere teleop")
                    break

                elif key == ord('c'):
                    self.capture_image()

                elif key == ord('v'):
                    if not self.recording:
                        self.start_video_recording()
                        self.recording = True
                    else:
                        self.stop_video_recording()
                        self.recording = False

                # Rotation axes ('x', 'y', 'z') are defined in the world frame
                # The delta angle is applied in the camera frame
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
            print("[WARN] No pose available to save")
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


    def stop_video_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print("[INFO] Video recording stopped.")


    def robot_move_to_camera_pose(self, base, H_wd_cam_des, speed=None):
        H_wd_ee_des = H_wd_cam_des @ np.linalg.inv(tf_to_hom_mtx(self.EE_cam_tf))
        p_world = H_wd_ee_des[:3, 3]
        R_ee = H_wd_ee_des[:3, :3]
        r_wd, p_wd, y_wd = rotation_matrix_to_euler(R_ee)
        r_wd, p_wd, y_wd = np.degrees(r_wd), np.degrees(p_wd), np.degrees(y_wd)

        p_des_kinova = np.array([p_world[0], p_world[1], p_world[2], r_wd, p_wd, y_wd])
        move_tool_pose_absolute(base, p_des_kinova, speed=speed)


if __name__ == "__main__":
    robot = MoveRobot(save_dir='/workspaces/isaac_ros-dev/src/proj_farmhand/proj_microscope_sim/data')
    robot.teleop_on_sphere()