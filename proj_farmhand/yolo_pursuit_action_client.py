import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from gen3_action_interfaces.action import YoloPursuit
import cv2
import numpy as np

class YoloVisualServoActionClient(Node):
    def __init__(self):
        super().__init__('yolo_visual_servo_action_client')
        self.get_logger().info('YoloVisualServoActionClient started, waiting for action server...')
        self._action_client = ActionClient(self, YoloPursuit, 'gen3_action/yolo_pursuit')

    def send_goal(self, percent_frame_height):
        goal_msg = YoloPursuit.Goal()
        goal_msg.des_yolo_diag = percent_frame_height
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


    
def main(args=None, percent_frame_height = 0.9):
    rclpy.init(args=args)
    action_client = YoloVisualServoActionClient()

    action_client.send_goal(percent_frame_height)
    rclpy.spin(action_client)
    future = action_client._get_result_future
    result = future.result().result
    print("Result: ", result)
    action_client.destroy_node()

if __name__ == '__main__':
    main()