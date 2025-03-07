import cv2
from PIL import Image
import os, sys
import numpy as np

# Get the absolute path of the current script
repo_root = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
sys.path.append(repo_root)
from Strawberry_Plant_Detection.detect import detect_boxes_only
# source ros2_kinova_ws/install/setup.bash before running this script
from gen3_7dof.tool_box import get_endoscope_tf_from_yaml, tf_to_hom_mtx
from gen3_7dof.tool_box import getRotMtx, R2rot
from gen3_7dof.tool_box import TCPArguments
from gen3_7dof.utilities import DeviceConnection

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2


class YoloVisualServoActionServer(Node):
    # Will make into action server later
    def __init__(self, base):
        super().__init__('yolo_visual_servo_action_server')
        ## Kortex API declarations
        self.base = base

        # Make sure the arm is in Single Level Servoing mode (high-level mode)
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        
        # kortex twist control mode
        self.command = Base_pb2.TwistCommand()
        # note that twist is naturally in tool frame, but this conversion made things a bit easy
        self.command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        self.command.duration = 0

        # initialize command storage
        self.twist = self.command.twist
        self.twist.linear_x = 0.0 
        self.twist.linear_y = 0.0
        self.twist.linear_z = 0.0
        self.twist.angular_x = 0.0
        self.twist.angular_y = 0.0
        self.twist.angular_z = 0.0 

        ## Control Parameters
        # assign initial desired pose as current pose
        self.desired_pose = base.GetMeasuredCartesianPose()
        self.R_d = getRotMtx(self.desired_pose)
        EE_endo_tf = get_endoscope_tf_from_yaml()
        self.EE_endo_mtx = tf_to_hom_mtx(EE_endo_tf)

        # control constants
        self.max_vel = 0.5-0.1  # m/s  0.5 is max
        #max_vel = 0.1
        self.max_w = 70.0  # ~ 50 deg/s
        self.kp_pos = 2.5
        self.kp_ang = 4.0
        
        # gains for IBVS
        self.percent_frame_height = 0.9 # note that this is the diagonal length of the box

        self.xy_P_gain = 0.7
        self.z_P_gain = 0.9

        self.prev_pos_diff_endo = np.zeros(3)
        self.xy_D_gain = 1.0
        self.z_D_gain = 1.5
        
        self.dcc_range = self.max_vel / (self.kp_pos * 2)  # dcc_range should be smaller than max_vel/kp_pos
        self.ang_dcc_range = self.max_w / (self.kp_ang * 6)

        self.eps_pos = 0.001  # convergence criterion
        self.eps_ang = 0.01

        self.no_flower_count = 0

        ## ROS2 declarations
        self.subscription = self.create_subscription(
            Image,
            'endoscope/resize/image',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()


    def IBVS_pos_only(self, box_cx, box_cy, box_length, des_x, des_y, des_length):  
        current_pose = self.base.GetMeasuredCartesianPose()
        R = getRotMtx(current_pose)
        
        # camera frame error (pixels) to error (meters)
        focal_length = 474.2788 # in pixels
        desired_z = 0.1
        pos_diff_endo = np.array([(box_cx-des_x)/focal_length*desired_z,
                            (box_cy-des_y)/focal_length*desired_z,
                            (des_length-box_length)/2/focal_length*desired_z])
        pos_error = pos_diff_endo * np.array([self.xy_P_gain, self.xy_P_gain, self.z_P_gain]) # element-wise multiplication
        vel_error = (pos_error - self.prev_pos_diff_endo) * np.array([self.xy_D_gain, self.xy_D_gain, self.z_D_gain])
        pos_diff_endo_composed = pos_error + vel_error
        
        self.prev_pos_diff_endo = pos_diff_endo

        # reducing the pos error (in BASE (world) frame!!)
        R_world_endo = R @ self.EE_endo_mtx[:3,:3]
        pos_diff = R_world_endo @ pos_diff_endo_composed

        # print("pos_diff_endo: ", pos_diff_endo)
        # print("pos_diff: ", pos_diff)
        pos_diff_norm = np.linalg.norm(pos_diff)+1e-5
        v_temp = self.max_vel * pos_diff/pos_diff_norm
        
        # reducing ang error using ER = RRd^T
        ER = R @ self.R_d.T
        # frobenius norm of matrix squre root of ER or eR2
        k,theta = R2rot(ER)
        k=np.array(k)
        eR2=-np.sin(theta/2)*k * 180 / np.pi  # not the best name but works fine
        
        eR2_norm = np.linalg.norm(eR2)+1e-5
        w_temp = self.max_w * eR2/eR2_norm

        # print(pos_diff_norm, eR2_norm)

        reached = pos_diff_norm < self.eps_pos and eR2_norm < self.eps_ang
        
        if reached:
            # print("Goal Pose reached")
            self.get_logger().info('Goal Pose reached')
            self.base.Stop()
            self.destroy_node()
        else: 
            # print the current error
            self.get_logger().info('pos_diff_norm: %f, eR2_norm: %f' % (pos_diff_norm, eR2_norm))
            # go in max vel when outside dcc_range
            if pos_diff_norm < self.dcc_range:
                v = self.kp_pos * pos_diff
            else:
                v = v_temp
    
            if eR2_norm < self.ang_dcc_range:
                kR = self.kp_ang*np.eye(3)
                w = kR @ eR2 
            else:
                w = w_temp
                
            self.twist.linear_x = v[0]
            self.twist.linear_y = v[1] 
            self.twist.linear_z = v[2]
            self.twist.angular_x = w[0]
            self.twist.angular_y = w[1]
            self.twist.angular_z = w[2]
            self.base.SendTwistCommand(self.command)


    def destroy_node(self):
        super().destroy_node()
        rclpy.shutdown()

    
    def get_largest_flower_box_center_and_length(self, boxes):
        # input: boxes (object detection results) (in device)
        # output: center_x, center_y, length (as numpy)
        max_area = 0
        flower_box = None

        for i in range(len(boxes.cls)):
            if boxes.cls[i] == 0:
                box = boxes.xyxy[i]
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > max_area:
                    max_area = area
                    flower_box = box

        # get the center pixel of the largest "flower" box
        if flower_box is not None:
            flower_box = flower_box.cpu().numpy()
            center_x = (flower_box[0] + flower_box[2]) / 2
            center_y = (flower_box[1] + flower_box[3]) / 2
            diag = np.sqrt((flower_box[2] - flower_box[0])**2 + (flower_box[3] - flower_box[1])**2)
            # length = max(flower_box[2] - flower_box[0], flower_box[3] - flower_box[1])
            return center_x, center_y, diag
        else:
            return None, None, None

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        boxes = detect_boxes_only(frame, confidence=0.7)
        # get the largest "flower" box center pixel
        center_x, center_y, length = self.get_largest_flower_box_center_and_length(boxes)
        if center_x is not None:
            self.no_flower_count = 0
            # get desired center from frame
            des_x, des_y = frame.shape[1] / 2, frame.shape[0] / 2
            des_length = self.percent_frame_height*frame.shape[0] # % of the frame height'

            self.IBVS_pos_only(center_x, center_y, length, des_x, des_y, des_length)
        else:
            self.no_flower_count += 1
            if self.no_flower_count > 10:
                self.base.Stop()
                self.get_logger().info('No flower detected for 10 frames. Stopped robot.')
                self.no_flower_count = 0
                # self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        # Create connection services
        base = BaseClient(router)
        robot_controller = YoloVisualServoActionServer(base)
        rclpy.spin(robot_controller)
    
    # robot_controller.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()