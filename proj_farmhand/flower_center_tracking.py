# take microscope image return flower center loc and ellipse properties
# Auther: Chuizheng Kong 
# Created on: 08/17/2024

import numpy as np
import cv2
import time

from gen3_7dof.tool_box import getRotMtx, R2rot, euler_to_rotation_matrix

storage_pt = np.zeros((10000, 3))
count = -1

def get_flower_center(frame, debug=False, show=False, REAL_FLOWER=False):
    ht, wd = frame.shape[0:2]
    center_xy = (int(wd/2), int(ht/2))
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    if not REAL_FLOWER:
        greenLower = (15, 230, 100)
        greenUpper = (25, 255, 220)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = center_xy  # when no ball find, center is just frame
    radius_list = np.array([1.0,1.0])

    if debug:
        global storage_pt
        global count
        print('screen center hsv', hsv[center_xy[1], center_xy[0],:])
        cv2.drawContours(frame, cnts, -1, (0,0,0), 4)

    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 5:
            return center, radius_list, frame
        # ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        ellipse = cv2.fitEllipse(cnt)
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        radius_a = ellipse[1][0] / 2
        radius_b = ellipse[1][1] / 2
        radius_list = np.sort([radius_a, radius_b])

        if radius_list[0] > 10:
            if show:
                cv2.ellipse(frame, ellipse, (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                temp_text1 = 'radius_a:' + str(int(radius_a))
                temp_text2 = 'radius_b:' + str(int(radius_b))
                frame = cv2.putText(frame, temp_text1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                frame = cv2.putText(frame, temp_text2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # print(hsv[center[1], center[0], :])
            if debug:
                count += 1
                storage_pt[count] = hsv[center[1], center[0], :]
                if count >= 10:
                    print('max H: {}, min H: {}, mean H:{} '.format(max(storage_pt[:count, 0]), min(storage_pt[:count, 0]), np.mean(storage_pt[:count, 0])))
                    print('max S: {}, min S: {}, mean S:{} '.format(max(storage_pt[:count, 1]), min(storage_pt[:count, 1]), np.mean(storage_pt[:count, 1])))
                    print('max V: {}, min V: {}, mean V:{} '.format(max(storage_pt[:count, 2]), min(storage_pt[:count, 2]), np.mean(storage_pt[:count, 2])))
                    print('\n')
                if count >= 10000:
                    storage_pt = np.zeros((10000, 3))
                    count = -1
    
    # move the center point to the frame origin same orientation as picture frame
    center = (center[0] - center_xy[0], center[1] - center_xy[1])
    return center, radius_list, frame


def calc_xy_plane_pose_error(flower_center, flower_radii):
    ## this function takes the flower ellipse properties and calculates the one step error in yz plane

    # # gains for the pursuit controller
    # kp_pos = 0.01 # need to be tuned, from pixel error to m/s
    # kp_ang = 0.1 # need to be tuned, from propotional error to deg/s

    # dcc_range = max_vel / (2 * kp_pos) # dcc_range should be smaller than max_vel / kp_pos
    # ang_dcc_range = max_w / (6 * kp_ang) # dcc_range should be smaller than max_vel / kp_ang

    # eps_pos = 5 # pixels
    # eps_ang = 0.2 # 1-r1/r2


    # based on the center of the ellipse, robot will move in the xy plane to center the ellipse
    pix_error_micro = np.array([flower_center[0], flower_center[1], 0])

    # based on the radio of the ellipse radii, robot will rotate it's fork around the +x axis
    # the ratio just has to meet some number around say 0.5 
    # as the jig can help certain degrees of rotation
    r1, r2 = flower_radii
    ang_error = (1 - r1/r2 )*np.pi/2 # due to the sorting, r1 < r2, mapping between 0 and pi/2

    return pix_error_micro, ang_error

    


if __name__ == '__main__':
    cap = cv2.VideoCapture(4,cv2.CAP_V4L2)
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        center, radius_list, frame = get_flower_center(frame, debug=True, show=True)
        pix_error_micro, ang_error = calc_xy_plane_pose_error(center, radius_list)
        print('pix_error_micro: ', pix_error_micro, 'ang_error: ', ang_error)
        print(center)
        frame = cv2.resize(frame, (540, 960))
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()