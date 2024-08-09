import cv2
import os, sys

# Get the absolute path of the current script
repo_root = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
sys.path.append(repo_root)
# from Strawberry_Plant_Detection.detect import track
from proj_farmhand.RS_tool_box import detect_sahi, draw_sahi_boxes


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class RealSenseSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_subscriber')
        self.color_sub = self.create_subscription(
            Image,
            'rs_on_link1/color/image_raw',
            self.image_callback,
            10)
        self.color_sub  # prevent unused variable warning
        self.aligned_depth_sub = self.create_subscription(
            Image,
            'rs_on_link1/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            'rs_on_link1/color/camera_info',
            self.camera_info_callback,
            10)
        self.camera_info_sub  # prevent unused variable warning


        self.aligned_depth_sub
        self.bridge = CvBridge()
        self.depth_img = None
        self.intrinsics = None

    def image_callback(self, msg):
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
        
        # frame = cv2.resize(frame, (w*2, h*2))
        # annotated_image, boxes = track(frame)
        # annotated_image, boxes, num_flowers, num_stamen = detect(frame)
        
        # When Slice Height and Width are None, the size will be determined autonomously
        pred_list = detect_sahi(frame,
                                slice_height=h//4,
                                slice_width=w//4)
        annotated_image, flower_centers_in_cam_frame = draw_sahi_boxes(frame, pred_list, self.depth_img, self.intrinsics)
        # make frame twice as large
        annotated_image = cv2.resize(annotated_image, (int(w/1.5), int(h/1.5)))
        cv2.imshow("Object Detection", annotated_image)
        cv2.waitKey(1)
        


    def depth_callback(self, msg):
        depth_img_hori = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # rotate 90 degrees clockwise
        self.depth_img = cv2.rotate(depth_img_hori, cv2.ROTATE_90_CLOCKWISE)
        # print('depth image received')

    def camera_info_callback(self, msg):
        self.intrinsics = msg


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = RealSenseSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    # img = cv2.imread("rs_img.jpg")
    # annotated_image, pred_list = detect_sahi(img)
    # cv2.imshow("Object Detection", annotated_image)
    # cv2.waitKey(0)