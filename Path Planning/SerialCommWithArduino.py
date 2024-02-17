# Importing Libraries
import serial
import time

# port changes every time you reconnect arduino
arduino = serial.Serial(port='/dev/cu.usbmodem1412401', baudrate=115200, timeout=.1) 

# def write_read(x):
#     arduino.write(bytes(x, 'utf-8'))
#     time.sleep(0.1)
#     data = arduino.readline()
#     return data

def startGrip():
    gripNum = 1
    while True:
        time.sleep(0.2)
        arduino.write(bytes(str(gripNum), 'utf-8'))
        time.sleep(0.1)
        data = arduino.readline().decode('utf-8') 
        # print("Data read from Arduino: ", data) 
        if data == '2':
            print("=======STARTED GRIPPING===========.")
            break

def finishGrip():
    finishGripNum = 3
    while True:
        time.sleep(0.2)
        arduino.write(bytes(str(finishGripNum), 'utf-8'))
        time.sleep(0.1)
        data = arduino.readline().decode('utf-8') 
        # print("Data read from Arduino: ", data) 
        if data == '4':
            print("=======FINISHED GRIPPING===========.")
            break

def ungrip():
    ungripNum = 5
    while True:
        time.sleep(0.2)
        arduino.write(bytes(str(ungripNum), 'utf-8'))
        time.sleep(0.1)
        data = arduino.readline().decode('utf-8') 
        # print("Data read from Arduino: ", data) 
        if data == '6':
            print("=======FINISHED UNGRIPPING===========.")
            break

# print("Start")
# startGrip()
# time.sleep(2)
# print("Finishing Gripping")
# finishGrip()
# time.sleep(2)
# ungrip()
# print("Ungripped")