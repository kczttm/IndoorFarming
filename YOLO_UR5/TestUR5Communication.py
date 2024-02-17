import socket
import time

# Cartesian tool pose (X, Y, Z, Roll, Pitch, Yaw).  Note: units are meter and rad. With respect to tool wrist frame (currently)
HOST = "169.254.14.134"  # Laptop (server) IP address, UR5 socket code on teach pendant should also have this IP address set to open. 
PORT = 30002

# Communicate with UR5
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))  # Bind to the port.
s.listen(5)  # Wait for UR5 (client) connection.
c, addr = s.accept()  # Establish connection with client.

try:
    print("In try branch.")
    msg = c.recv(1024).decode()  # Receive message from UR5.
    if msg == 'UR5_is_asking_for_data':
        print(msg)
        pose = [0, 0, 0, 0, 0, 0]
        values = str(pose[0]) + ", " + str(pose[1]) + ", " + str(pose[2]) + \
        ", " + str(pose[3]) + ", " + str(pose[4]) + ", " + str(pose[5])
        poseString = "(" + values + ")"
        print("Sending pose.")
        c.send(poseString.encode())
        c.close()
        s.close()

except socket.error as socketerror:
    print("Error occured.")