import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from gen3_action_interfaces.action import RealSenseFlowerPoses
import cv2
import numpy as np
np.set_printoptions(suppress=True)

class RealSenseFlowerPosesActionClient(Node):
    def __init__(self):
        super().__init__('rs_flower_poses_action_client')
        self.get_logger().info('RealSenseFlowerPosesActionClient started, waiting for action server...')
        self._action_client = ActionClient(self, RealSenseFlowerPoses, 'realsense_action/flower_poses')

    def send_goal(self, sahi_n_slices):
        goal_msg = RealSenseFlowerPoses.Goal()
        goal_msg.sahi_n_slices = sahi_n_slices
        self._action_client.wait_for_server()
        
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback received: {feedback_msg.feedback}')

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result 

        #shutdown after the result is received
        rclpy.shutdown() 


    
def main(args=None, sahi_n_slices = 4):
    rclpy.init(args=args)
    action_client = RealSenseFlowerPosesActionClient()

    action_client.send_goal(sahi_n_slices)
    rclpy.spin(action_client)
    future = action_client._get_result_future
    result = future.result().result
    action_client.destroy_node()
    # print("Flower Poses: ", np.array(result.result))
    if len(result.result) == 0:
        print("No flower data collected")
        return np.array([])
    else:
        return np.array(result.result).reshape(-1, 3)

if __name__ == '__main__':
    main()