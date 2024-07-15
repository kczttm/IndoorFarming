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
    # Number value - time if R/E, lin actuator position if P (min 0 max 630)
    # servo position if Z
    ReceivedString = write_read(SerialObj, "<ABC 123;>")
    print(ReceivedString)
    
    return SerialObj

# autofocus code from microscope company
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

def auto_focus(SerialObj, cam_id=4, predefined_pos=0):
    time.sleep(1)
    # declare loop variables
    data_counter = 0
    focus_score = 0
    focus_score_max = 30 # for 1920x1080
    fc_temp = 0
    fc_sum = 0

    # fine level adjustment flags
    s_coarse = True

    # open camera
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()  

    # pos_str = "<P " + str(predefined_pos) + ";>"
    # write_only(SerialObj, pos_str)
    # print("moving to predefined position")
    # s_coarse = False

    # autofocus loop
    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # crop out the center of the frame
        h, w = frame.shape[:2]
        h_des = 400
        w_des = 400
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

            # microscope coarse-fine adjustment
            if s_coarse:
                if focus_score < focus_score_max:
                    # moving down fast
                    write_only(SerialObj, "<R 100;>")
                elif focus_score >= focus_score_max and focus_score < (focus_score_max * 2.2):
                    # overshooted
                    # enter fine adjustment mode
                    focus_score_max = focus_score - focus_score * 0.2
                    s_coarse = False
                else:
                    # pos_str = "<P " + str(0) + ";>"
                    # write_only(SerialObj, pos_str)
                    # print("moving to max position")
                    # s_coarse = False
                    write_only(SerialObj, "<R 100;>")
            else:
                if focus_score < focus_score_max:
                    print("Moving up slow")
                    # moving down slow
                    write_only(SerialObj, "<E 10;>")
                else:
                    # focus achieved
                    print("Focus Achieved")
                    # s_coarse = True
        else:
            fc_sum += fc_temp
            data_counter += 1
        
        # display frame
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow('Microscope Autofocusing', frame)

        usr_key = cv2.waitKey(1)
        if usr_key == ord('q'):
            break  

    cap.release()
    cv2.destroyAllWindows()
    write_only(SerialObj, "<P 630;>") 

        


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
