# source the ros2_kinova_ws/install/setup.bash before running this script
import os, sys
import cv2

script_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
sys.path.append(repo_root)

from gen3_7dof.take_pictures_action_client import main as TakePicturesActionClient
from RAFT_tool_box import load_model, inference, get_viz

DEBUG = False

def take_pictures():
    spacing = 0.005
    pictures = TakePicturesActionClient(spacing=spacing)
    return pictures[0], pictures[-1]

def main():
    RAFT_model = load_model()
    if DEBUG:
        img1_name = "Flower_on_vine_1_5mm_offset.png"
        img2_name = "Flower_on_vine_2_5mm_offset.png"
        frame1 = cv2.imread(os.path.join(script_dir, img1_name))
        frame2 = cv2.imread(os.path.join(script_dir, img2_name))
    else:
        frame1, frame2 = take_pictures()

    flow_iters = inference(RAFT_model, frame1, frame2, iters=50, test_mode=False) 

    # Display the flow with cv2
    flow_viz = cv2.cvtColor(get_viz(flow_iters[-1]), cv2.COLOR_RGB2BGR)
    cv2.imshow("Flow Visualization", flow_viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()