import cv2
import numpy as np
import serial
import time
# import imutils

def arduino_connect(port='/dev/ttyUSB0'):
    try :
        SerialObj = serial.Serial(port)
    except serial.serialutil.SerialException as e:
        print('An Error Occurred: ', e)
    else:
        print('Arduino Connected')
    SerialObj.baudrate = 9600  # set Baud rate to 9600
    SerialObj.bytesize = 8     # Number of data bits = 8
    SerialObj.parity   ='N'    # No parity
    SerialObj.stopbits = 1     # Number of Stop bits = 1
    SerialObj.timeout  = None  # Setting timeouts: None = waits forever
    time.sleep(3)              # timing for Arduino

    # Expected command format: Expected format: "<ABC 123;>"
    # P - positional
    # R - retract
    # E - extend
    # Z - zoom
    # G - get potentiometer value
    # Number value - time if R/E, lin actuator position if P (min 0 max 635)
    # servo position if Z, number does not matter for G
    ReceivedString = write_read(SerialObj, "<ABC 123;>")
    print(ReceivedString)
    
    return SerialObj

# autofocus score based on Zaber microscope example
def calculate_focus_score(image, blur=9):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # image_filtered = cv2.GaussianBlur(gray, (5, 5), 0)
    # laplacian = cv2.Laplacian(image_filtered, cv2.CV_64F)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    focus_score = laplacian.var()
    return focus_score

def write_read(SerialObj, input):
    SerialObj.write(bytes(input, 'utf-8'))
    time.sleep(0.05)
    data = SerialObj.readline()
    return data

def write_only(SerialObj, input):
    SerialObj.write(bytes(input, 'utf-8'))

# construct a command in expected format
def motor_command(SerialObj, txt, val):
    str_val = str(val)
    cmd = '<' + txt + ' ' + str_val + ';>'
    data = write_read(SerialObj, cmd)
    return data


def auto_focus(SerialObj, cam_id=4, predefined_pos=0):
    time.sleep(1)
    # declare loop variables
    data_counter = 0
    focus_score = 0
    focus_score_max = 100 # for 200 by 200 crop area
    fc_temp = 0
    fc_sum = 0

    # focus flags
    hold_zoom = False
    extend_flag = False
    retract_flag = True

    # open camera
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()  

    # autofocus loop
    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # crop out the center of the frame
        h, w = frame.shape[:2]
        h_des = 200
        w_des = 200
        h_start = int((h - h_des)/2)
        w_start = int((w - w_des)/2)
        cent_crop = frame[h_start:h_start+h_des, w_start:w_start+w_des]

        # calculate focus score
        fc_temp = calculate_focus_score(cent_crop)

        # draw rectangle on the frame
        cv2.rectangle(frame, (w_start, h_start), (w_start+w_des, h_start+h_des), (0, 255, 0), 2)

        # get the average focus score of 10 frames for stability
        if data_counter >= 10:
            focus_score = fc_sum / data_counter
            print("Focus Score: ", focus_score, " Focus Score Max: ", focus_score_max)
            fc_sum = 0
            data_counter = 0

            # Microscope zoom focus decision tree
            focus_timer += 1
            potval = motor_command(SerialObj, 'G', 0)
            if not hold_zoom and not focus_wait:
                zoom_e = focus_score_max - focus_score
                # print(zoom_e)
                if zoom_e < 0:
                    zoom_e = 0
                kp = 1 / 40 # max zoom val (180 deg) / max variance val (in the thousands)
                e_kp = zoom_e * kp
                # print(e_kp)
                if focus_score < focus_score_max:
                    if retract_flag: # determine which direction based on flag
                        if zoom_val < 180:
                            zoom_val += e_kp
                        if zoom_val >= 180:
                            zoom_val = 180
                    if extend_flag:
                        if zoom_val > 0:
                            zoom_val -= e_kp
                        if zoom_val <= 0:
                            zoom_val = 0
                    motor_command(SerialObj, 'Z', zoom_val)
                else: # hold zoom and save values
                    focus_score_max = focus_score
                    hold_zoom = True
                    print("Focus achieved!")
            if focus_timer > 2:
                focus_wait = False
        else:
            fc_sum += fc_temp
            data_counter += 1
        
        # display frame
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow('Microscope Autofocusing', frame)
        
        usr_key = cv2.waitKey(1)
        match usr_key:
            case ord('q'):
                break
            case ord('p'):
                new_pos = input("Input position: ")
                # create command for position
                if new_pos.isnumeric():
                    new_pos = int(new_pos)
                else:
                    new_pos = 635
                potval = int(motor_command(SerialObj, 'G', 0))
                # print(new_pos)
                motor_command(SerialObj, 'P', new_pos)
                # determine whether extending or retracting
                if new_pos > potval:
                    extend_flag = True
                    retract_flag = False
                elif new_pos < potval:
                    extend_flag = False
                    retract_flag = True
                else: 
                    extend_flag = False
                    retract_flag = False

                focus_score_max = 100
                hold_zoom = False
                focus_wait = True
                focus_timer = 0
            case _:
                continue

    cap.release()
    cv2.destroyAllWindows()
    motor_command(SerialObj, 'P', 635)

if __name__ == '__main__':
    SerialObj = arduino_connect()
    time.sleep(2)
    auto_focus(SerialObj)
    # write_only(SerialObj, "<P 0;>")
    # time.sleep(2)
    # write_only(SerialObj, "<Z 180;>")
    # time.sleep(2)
    # write_only(SerialObj, "<Z 0;>")
    # time.sleep(2)
    # write_only(SerialObj, "<P 630;>")
    # time.sleep(2)
    SerialObj.close()
    print('Serial Port Closed')
