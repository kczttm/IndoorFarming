import socket
import time
import cv2
import numpy as np
import os


def build_pose_string(pose):
    # pose is a list with length of 6
    # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
    values = str(pose[0]) + ", " + str(pose[1]) + ", " + str(pose[2]) + \
        ", " + str(pose[3]) + ", " + str(pose[4]) + ", " + str(pose[5])
    poseString = "(" + values + ")"
    return poseString


def move_robot(pose):
    # Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
    HOST = "169.254.14.134"  # Laptop (server) IP address.
    PORT = 30002

    # Communicate with UR5
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))  # Bind to the port.
    s.listen(5)  # Wait for UR5 (client) connection.
    c, addr = s.accept()  # Establish connection with client.

    try:
        msg = c.recv(1024).decode()  # Receive message from UR5.
        if msg == 'UR3_is_asking_for_data':
            poseString = build_pose_string(pose)
            c.send(poseString.encode())
            c.close()
            s.close()

    except socket.error as socketerror:
            print("Error occured.")

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

os.chdir("./new_test_img")

move = np.zeros((6,))
move[2] = move[2] - 0.06
move_robot(move)

for i in range(35):
    move = np.zeros((6,))
    move[2] = 0.005

    move_robot(move)
    curr_image = None

    for _ in range(5):
        result, curr_image = cam.read() 
    image_name = "img_"+str(i)+".jpg"
    cv2.imwrite(image_name, curr_image)

cam.release()