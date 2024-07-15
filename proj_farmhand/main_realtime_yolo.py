from re import T
import cv2
from PIL import Image
import os, sys

# Get the absolute path of the current script
repo_root = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
sys.path.append(repo_root)
from Strawberry_Plant_Detection.detect import detect

USE_ROS = True
TEST_IMG = False

if USE_ROS:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    class ImageSubscriber(Node):
        def __init__(self):
            super().__init__('image_subscriber')
            self.subscription = self.create_subscription(
                Image,
                'endoscope/resize/image',
                self.image_callback,
                10)
            self.subscription  # prevent unused variable warning
            self.bridge = CvBridge()

        def image_callback(self, msg):
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            annotated_image, boxes, num_flowers, num_stamen = detect(frame)
            # make frame twice as large
            annotated_image = cv2.resize(annotated_image, (annotated_image.shape[1]*2, annotated_image.shape[0]*2))
            cv2.imshow("Object Detection", annotated_image)
            cv2.waitKey(1)
    
    def main(args=None):
        rclpy.init(args=args)
        image_subscriber = ImageSubscriber()
        rclpy.spin(image_subscriber)
        rclpy.shutdown()

    if __name__ == '__main__':
        main()


elif TEST_IMG:
    # Load the image
    curdir = os.path.dirname(os.path.realpath(__file__))
    image_path = curdir+"/test2.png"
    print(image_path)
    frame = cv2.imread(image_path)

    annotated_image, boxes, num_flowers, num_stamen = detect(frame)
    # Display the frame
    cv2.imshow("Object Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
else:
    # # Load the pre-trained YOLOv3 model
    # model = detection.yolo_v3(pretrained=True)

    # # Set the model to evaluation mode
    # model.eval()

    # # Load the COCO class labels
    # with open("coco_labels.txt", "r") as f:
    #     labels = f.read().splitlines()

    # # Define the transformation to apply to the input image
    # transform = transforms.Compose([
    #     transforms.Resize((416, 416)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # Initialize the video capture
    cap = cv2.VideoCapture(2,cv2.CAP_V4L2)

    if not cap.isOpened():
            print("Cannot open camera")
            sys.exit()

    while True:
        # Read the frame from the video capture
        ret, frame = cap.read()
        if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
        
        # stricktly not converting!!
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB because openCV uses BGR

        # yolov8 prediction
        annotated_image, boxes, num_flowers, num_stamen = detect(frame)
        # make frame twice as large
        annotated_image = cv2.resize(annotated_image, (annotated_image.shape[1]*2, annotated_image.shape[0]*2))
          
        # Display the frame
        cv2.imshow("Object Detection", annotated_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # if cv2.waitKey(1) & 0xFF == ord('s'):
        #     cv2.imwrite('endo_image.png', frame)
        #     break
        

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()