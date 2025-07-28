import os
import cv2
import math
import numpy as np
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from proj_microscope_sim.cam_pose import get_world_EE_HomoMtx
from proj_farmhand.main_full_pipeline import robot_pose_estimation

from geometry_msgs.msg import TransformStamped
from gen3_7dof.tool_box import rotation_matrix_to_euler, tf_to_hom_mtx, move_tool_pose_absolute, TCPArguments
from gen3_7dof.utilities import DeviceConnection
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

import matplotlib.pyplot as plt


class MoveRobot(Node):
    def __init__(self, save_dir, device_id=3):
        super().__init__('teleop_camera')

        self.device_id = device_id

        # Generate timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"session_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)


        # Initialize camera; device_id = 2 if using laptop
        # try:
        #     self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        #     if not self.cap.isOpened():
        #         raise RuntimeError(f"[ERROR] Failed to open camera device {self.device_id}")
        #     print(f"[INFO] Camera {self.device_id} opened successfully")
        
        # except Exception as e:
        #     print(f"[WARN] Camera initialization failed: {e}")
        #     self.cap = None

        
        self.bridge = CvBridge()
        self.latest_frame = None
        self.image_sub = self.create_subscription(
            Image,
            '/endoscope/resize/image',
            self.image_callback,
            10
        )

        
        # Initialize a cv2.VideoWriter object
        self.video_writer = None
        self.recording = False


        self.EE_cam_tf = self.get_EE_camera_tf()
        self.image_counter = 0
        self.pose_log = []
        self.captured_poses = []

    
    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")


    def __del__(self):
        # try:
        #     if hasattr(self, 'cap') and self.cap:
        #         self.cap.release()
        #         print("[INFO] Camera released")
        # except Exception:
        #     pass

        try:
            cv2.destroyAllWindows()
            print("[INFO] OpenCV windows closed")
        except Exception:
            pass

    
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

    
    def robot_move_to_camera_pose(self, base, H_wd_cam_des, speed=None):
        H_wd_ee_des = H_wd_cam_des @ np.linalg.inv(tf_to_hom_mtx(self.EE_cam_tf))
        p_world = H_wd_ee_des[:3, 3]
        R_ee = H_wd_ee_des[:3, :3]
        r_wd, p_wd, y_wd = rotation_matrix_to_euler(R_ee)
        r_wd, p_wd, y_wd = np.degrees(r_wd), np.degrees(p_wd), np.degrees(y_wd)

        p_des_kinova = np.array([p_world[0], p_world[1], p_world[2], r_wd, p_wd, y_wd])
        move_tool_pose_absolute(base, p_des_kinova, speed=speed)


    def estimate_flower_center_and_radius(self, base):
        """
        Automatically detect the flower center and radius using YOLO and ICP
        Sets self.center and self.radius accordingly
        """

        # Estimate flower position in camera frame
        # H_cam_flower, *_ = robot_pose_estimation(visualize=False, real_flower=False)
        H_cam_flower, *_ = robot_pose_estimation(parent_node=self, visualize=False, real_flower=False)

        # Get current EE and compute camera pose
        H_wd_ee = get_world_EE_HomoMtx(base)
        H_wd_cam = H_wd_ee @ tf_to_hom_mtx(self.EE_cam_tf)

        # Get flower center in world frame
        flower_in_world = H_wd_cam @ np.append(H_cam_flower[:3, 3], 1.0)
        self.center = flower_in_world[:3]

        # Compute radius (camera to flower distance)
        cam_pos = H_wd_cam[:3, 3]
        self.radius = np.linalg.norm(self.center - cam_pos)
        
        print(f"[INFO] Flower center (world): {self.center}")
        print(f"[INFO] Computed radius: {self.radius:.4f} m")

        self.pose_log = [H_wd_cam.copy()]  # reset and save initial pose after estimating flower
    

    # TODO: super weired movement when using this function
    def auto_capture_sphere(self, base, speed=0.03):
        """
        Automatically move around a virtual sphere centered on the flower and capture images + poses
        """

        # Function to visualize camera poses on a sphere
        def visualize_camera_poses(poses, center, draw_axes=False):
            """
            Visualize camera poses and their orientation vectors in 3D

            Args:
                poses (List[np.ndarray]): List of 4x4 camera-to-world matrices
                center (np.ndarray): The point all cameras look at (default: origin)
                show_axes (bool): Whether to draw local axes at each camera pose
            """

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            cam_positions = np.array([H[:3, 3] for H in poses])
            ax.scatter(*cam_positions.T, color='red', label='Camera Positions')

            for i, H in enumerate(poses):
                origin = H[:3, 3]
                forward = H[:3, 2]  # camera's +Z axis (pointing at center)

                # Draw forward direction only
                ax.quiver(origin[0], origin[1], origin[2],
                        forward[0], forward[1], forward[2],
                        length=0.05, color='blue', normalize=True)

                if draw_axes:
                    R = H[:3, :3]
                    colors = ['r', 'g', 'b']
                    for j in range(3):  # x, y, z
                        ax.quiver(origin[0], origin[1], origin[2],
                                R[0, j], R[1, j], R[2, j],
                                length=0.04, color=colors[j], normalize=True)


            # Draw object center
            ax.scatter(center[0], center[1], center[2], color='black', s=50, label='Flower Center')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Camera Poses')
            ax.legend()
            ax.set_box_aspect([1, 1, 1])  # Equal scaling
            plt.show()


        # Function to generate poses on a sphere
        def generate_poses(center, radius, num_arcs=4, points_per_arc=16):
            poses = []
            azimuths = np.linspace(0, 2 * np.pi, num_arcs, endpoint=False)
            # elevations = np.linspace(np.pi / 6, np.pi / 2, points_per_arc)  # 30° to 90°
            elevations = np.linspace(0, np.pi, points_per_arc)

            for phi in azimuths:
                for theta in elevations:
                    # Spherical to Cartesian
                    # Rotate around Y-axis (not good)
                    # x = radius * np.sin(theta) * np.cos(phi)
                    # y = radius * np.sin(theta) * np.sin(phi)
                    # z = radius * np.cos(theta)

                    # Rotate around base Z-axis
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta) * np.cos(phi)
                    z = radius * np.sin(theta) * np.sin(phi)

                    # Rotate around X-axis (not good)
                    # x = radius * np.sin(theta) * np.sin(phi)
                    # y = radius * np.sin(theta) * np.cos(phi)
                    # z = radius * np.cos(theta)

                    cam_pos = np.array([x, y, z]) + center

                    # Orientation: face the center
                    forward = (center - cam_pos)
                    forward /= np.linalg.norm(forward)

                    # Up vector = world Y
                    up = np.array([0, 1, 0])
                    right = np.cross(up, forward)
                    right /= np.linalg.norm(right)
                    up = np.cross(forward, right)

                    R = np.stack([right, up, forward], axis=1)

                    H = np.eye(4)
                    H[:3, :3] = R
                    H[:3, 3] = cam_pos
                    poses.append(H)

                    # Debugging prints
                    print(f"[DEBUG] Pose generated:")
                    print(f"  Cartesian Position: {cam_pos}")
                    print(f"  Orientation Matrix (R):\n{R}")

            return poses


        print("[INFO] Generating auto-capture poses on virtual sphere...")
        poses = generate_poses(self.center, self.radius, num_arcs=1, points_per_arc=16)
        visualize_camera_poses(poses, center=self.center, draw_axes=False)

        for i, H_wd_cam in enumerate(poses):
            print(f"[INFO] Moving to view {i+1}/{len(poses)}...")
            self.robot_move_to_camera_pose(base, H_wd_cam, speed=speed)
            rclpy.spin_once(self, timeout_sec=0.5)

            if self.latest_frame is not None:
                img_filename = os.path.join(self.save_dir, f"auto_{i:04d}.png")
                cv2.imwrite(img_filename, self.latest_frame)
                self.captured_poses.append(H_wd_cam.copy())
                print(f"[INFO] Captured {img_filename}")
            else:
                print(f"[WARN] Skipped capture {i} (no frame available)")

    
    def run_auto_capture_session(self, speed=0.03, filename="auto_poses_bounds.npy"):
        tcp_args = TCPArguments()
        with DeviceConnection.createTcpConnection(tcp_args) as router:
            base = BaseClient(router)
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            base.SetServoingMode(base_servo_mode)

            # Step 1: Estimate center + radius
            self.estimate_flower_center_and_radius(base)

            # Step 2: Automatically capture views
            self.auto_capture_sphere(base, speed=speed)

            # Step 3: Save LLFF-style poses
            self.save_pose_log(filename)
            print(f"[INFO] Auto-capture session complete. Saved to: {os.path.join(self.save_dir, filename)}")


    def teleop_on_sphere(self, pitch_step=0.5, yaw_step=0.5, speed=0.03):
        tcp_args = TCPArguments()
        with DeviceConnection.createTcpConnection(tcp_args) as router:
            base = BaseClient(router)
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            base.SetServoingMode(base_servo_mode)

            # warmup_frames = 30
            # for _ in range(warmup_frames):
            #     ret, frame = self.cap.read()
            #     if ret:
            #         msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            #         self.image_pub.publish(msg)
            #     rclpy.spin_once(self, timeout_sec=0.01)

            # Auto-sphere initialization
            self.estimate_flower_center_and_radius(base)

            while True:
                rclpy.spin_once(self, timeout_sec=0.01)
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    cv2.imshow("Sphere Teleop", frame)

                # ret, frame = self.cap.read()
                # if ret:
                #     cv2.imshow("Sphere Teleop", frame)
                #     msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                #     self.image_pub.publish(msg)

                    if self.recording and self.video_writer is not None:
                        self.video_writer.write(frame)


                key = cv2.waitKey(10) & 0xFF

                H_wd_ee = get_world_EE_HomoMtx(base)
                init_H_wd_cam = H_wd_ee @ tf_to_hom_mtx(self.EE_cam_tf)

                moved = False
                # H_cam_flower = np.array([[1, 0, 0, 0],
                #                         [0, 1, 0, 0],
                #                         [0, 0, 1, 0.1],  # +10 cm in Z
                #                         [0, 0, 0, 1]
                #                         ])
                # H_wd_flower = init_H_wd_cam @ H_cam_flower
                
                # H_wd_flower = np.eye(4)
                # H_wd_flower[:3, 3] = self.center

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
                    # H_wd_cam = self.pose_log[-1]
                    
                    # Rotate the camera pose around the flower center
                    T_to_center = np.eye(4)
                    # T_to_center[:3, 3] = -H_wd_flower[:3, 3]
                    T_to_center[:3, 3] = -self.center

                    T_from_center = np.eye(4)
                    # T_from_center[:3, 3] = H_wd_flower[:3, 3]
                    T_from_center[:3, 3] = self.center


                    H_wd_cam_des = T_from_center @ H_cam_delta @ T_to_center @ H_wd_cam

                    # Move robot once and log pose
                    self.robot_move_to_camera_pose(base, H_wd_cam_des, speed=speed)
                    self.pose_log.append(H_wd_cam_des.copy())


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


    def capture_image(self):
        # ret, frame = self.cap.read()
        frame = self.latest_frame
        if frame is None:
            print("[WARN] No image available")
            return

        if len(self.pose_log) == 0:
            print("[WARN] No pose available to save")
            return
        
        # if ret:
        # Save image
        img_filename = os.path.join(self.save_dir, f"img_{self.image_counter:04d}.png")
        cv2.imwrite(img_filename, frame)

        # Save pose
        self.captured_poses.append(self.pose_log[-1].copy())
        print(f"[INFO] Saved img_{self.image_counter:04d}.png and pose")
            
        self.image_counter += 1


    def start_video_recording(self, filename="output.avi", fps=10):
        # height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if self.latest_frame is None:
            raise RuntimeError("[ERROR] No image available yet to determine resolution.")

        height, width = self.latest_frame.shape[:2]

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

    
def main():
    rclpy.init()
    robot = MoveRobot(save_dir='/workspaces/isaac_ros-dev/src/proj_farmhand/proj_microscope_sim/data')
    # robot.teleop_on_sphere()
    robot.run_auto_capture_session(speed=0.03)
    robot.destroy_node()
    rclpy.shutdown()
