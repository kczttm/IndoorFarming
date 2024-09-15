import os, sys
import numpy as np
import time
import cv2

# Get the absolute path of the current script
repo_root = os.path.abspath(os.path.join(os.path.abspath(__file__), 
                                         os.pardir, os.pardir, os.pardir, os.pardir,
                                         "src", "proj_farmhand")) # ensure that it's not in build
# sys.path.append(repo_root)
exp_data_dir = os.path.join(repo_root, "ExperimentData")

from proj_farmhand.RS_tool_box import detect_sahi, draw_sahi_boxes

import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from gen3_action_interfaces.action import RealSenseFlowerPoses

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2

from gen3_7dof.tool_box import get_joint_angles, move_joints, get_realsense_on_link1_HomoMtx
from gen3_7dof.tool_box import TCPArguments
from gen3_7dof.utilities import DeviceConnection


class RealSenseFlowerPosesActionServer(Node):
    def __init__(self):
        super().__init__('rs_flower_poses_action_server')

        # Kortex API declarations
        self._tcp_args = TCPArguments()
        self.base = None
        self.init_joint_angles = None

        # Action server for pose_estimation in camera frame
        self._action_server = ActionServer(
            self,
            RealSenseFlowerPoses,
            'realsense_action/flower_poses',
            callback_group=ReentrantCallbackGroup(),
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback)
        self._job_active = False
        self._current_goal = None
        self.selected_flower_poses = np.empty((0,3))

        ## ROS2 declarations
        ###?????????? somehow by reordering the sub declarations, 
        ### the depth start working?? need to figure this out
        self.aligned_depth_sub = self.create_subscription(
            Image,
            'rs_on_link1/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10)
        self.aligned_depth_sub

        self.color_sub = self.create_subscription(
            Image,
            'rs_on_link1/color/image_raw',
            self.image_callback,
            10)
        self.color_sub  # prevent unused variable warning

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            'rs_on_link1/color/camera_info',
            self.camera_info_callback,
            10)
        self.camera_info_sub  # prevent unused variable warning

        self.bridge = CvBridge()

        # realsense collected data
        self.depth_img = None
        self.intrinsics = None
        self.sahi_n_slices = 4  # increase this number is target is further away
        self.overlap_ratio = 0.75


    def image_callback(self, msg):
        if not self._job_active:
            return
        if self.depth_img is None or self.intrinsics is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # if not self.took_img:
        #     cv2.imwrite("rs_img.jpg", frame)
        #     self.took_img = True
        # # roi
        # w,h = frame.shape[1], frame.shape[0]
        # frame = frame[h-w-100:h-100,:]

        w,h = frame.shape[1], frame.shape[0]
        H_wd_rs = get_realsense_on_link1_HomoMtx(self.base)
        
        # When Slice Height and Width are None, the size will be determined autonomously
        pred_list = detect_sahi(frame,
                                slice_height=h//self.sahi_n_slices,
                                slice_width=w//self.sahi_n_slices,
                                overlap_h_ratio=self.overlap_ratio,
                                overlap_w_ratio=self.overlap_ratio)
        annotated_image, flower_centers_in_world_frame = draw_sahi_boxes(frame, 
                                                                       pred_list, 
                                                                       self.depth_img, 
                                                                       self.intrinsics,
                                                                       H_wd_rs)
        # make frame twice as large
        annotated_image = cv2.resize(annotated_image, (int(w/1.5), int(h/1.5)))
        cv2.imshow("Object Detection", annotated_image)

        ##press s to collect the flower data, press q to quit
        key = cv2.waitKey(1)
        if key == ord('q'):
            self._job_active = False
            self._current_goal.publish_feedback(RealSenseFlowerPoses.Feedback(status='Job aborted'))
        if key == ord('s'):
            self._job_active = False
            # print(flower_centers_in_cam_frame)
            self.selected_flower_poses = flower_centers_in_world_frame
            self._current_goal.publish_feedback(RealSenseFlowerPoses.Feedback(status='Flower data collected'))
            # obtain the image name string if exists
            rs_storage_path = os.path.join(exp_data_dir, "rs_img_name_str.npy")
            if os.path.exists(rs_storage_path):
                # convert the numpy array to string
                img_name_str = str(np.load(rs_storage_path, allow_pickle=True))
                cv2.imwrite(img_name_str, annotated_image)



    def depth_callback(self, msg):
        depth_img_hori = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # rotate 90 degrees clockwise
        self.depth_img = cv2.rotate(depth_img_hori, cv2.ROTATE_90_CLOCKWISE)
        # print('depth image received')

    def camera_info_callback(self, msg):
        self.intrinsics = msg
        # print('camera info received')


    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT
    
    
    async def execute_callback(self, goal_handle):
        self._current_goal = goal_handle
        self.get_logger().info('Executing goal...')
        result = RealSenseFlowerPoses.Result()
        try:
            if goal_handle.request.sahi_n_slices is not None:
                self.sahi_n_slices = goal_handle.request.sahi_n_slices

            with DeviceConnection.createTcpConnection(self._tcp_args) as router:
                self.base = BaseClient(router)

                # Make sure the arm is in Single Level Servoing mode (high-level mode)
                base_servo_mode = Base_pb2.ServoingModeInformation()
                base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
                self.base.SetServoingMode(base_servo_mode)

                self.init_joint_angles = get_joint_angles(self.base)

                # rise the robot to take pictures
                joint_angles = self.init_joint_angles.copy()
                joint_angles[1] = 0
                joint_angles[5] = 65
                # move to the initial joint angles
                action_result = move_joints(self.base, joint_angles)

                self._job_active = True
                # need to test if this is the correct waiting function in async env
                # rclpy.spin_until_future_complete(self, self._current_goal.get_feedback())
                while rclpy.ok() and self._job_active:
                    time.sleep(0.01)
                
                action_result = move_joints(self.base, self.init_joint_angles)

        except Exception as e:
            self.get_logger().info('An error occurred: ' + str(e))
            self._job_active = False
        
        cv2.destroyAllWindows()
        

        if self.selected_flower_poses.size == 0:
            print("No flower data collected")
            result.result = self.selected_flower_poses.tolist()
            self._current_goal.abort()
        else:
            result.result = self.selected_flower_poses.flatten().tolist()
            self._current_goal.succeed()
        
        self._current_goal = None
        self.selected_flower_poses = np.empty((0,3))
        return result


def main(args=None):
    rclpy.init(args=args)
    rs_server = RealSenseFlowerPosesActionServer()
    
    executor = MultiThreadedExecutor()

    rclpy.spin(rs_server, executor=executor)
    rclpy.destroy_node(rs_server)
    print('Shutting down...')
    rclpy.shutdown()

if __name__ == '__main__':
    main()