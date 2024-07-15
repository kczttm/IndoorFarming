import numpy as np
import imutils
import cv2
import serial
import time

# Image parameters
width = 1920
height = 1080

# autofocus code from microscope company
def calculate_focus_score(image, blur=9):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # image_filtered = cv2.bilateralFilter(image, 9, 75, 75)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # laplacian = cv2.Laplacian(image, cv2.CV_64F)
    focus_score = laplacian.var()
    return focus_score

def write_read(input):
    SerialObj.write(bytes(input, 'utf-8'))
    time.sleep(0.05)
    data = SerialObj.readline()
    return data

def write_only(input):
    SerialObj.write(bytes(input, 'utf-8'))

# Check devices before running
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Object tracking variables
tracker = cv2.TrackerKCF_create()
r = None
crop_max = 150
cent_max = 15
data_timer_f = 0
c_cont_sum = 0
t_cont_sum = 0
cent_cont_sum = 0
potval = 0
total_collect = np.zeros(10)
s_coarse = True
s_fine = False
focus_mode = False

data_timer = 0

# Arduino Serial comms

try:
    SerialObj = serial.Serial('/dev/ttyUSB0') # open the Serial Port
except serial.SerialException as var : # var contains details of issue
    print('An Exception Occured')
    print('Exception Details-> ', var)
else:
    print('Serial Port Opened') 

SerialObj.baudrate = 9600  # set Baud rate to 9600
SerialObj.bytesize = 8     # Number of data bits = 8
SerialObj.parity   ='N'    # No parity
SerialObj.stopbits = 1     # Number of Stop bits = 1
SerialObj.timeout  = None  # Setting timeouts: None = waits forever
time.sleep(3)              # timing for Arduino
# SerialObj.write(b'A')

# Expected command format: Expected format: "<ABC 123;>"
# P - positional
# R - retract
# E - extend
# Z - zoom
# Number value - time if R/E, lin actuator position if P (min 0 max 630)
# servo position if Z
ReceivedString = write_read("<ABC 123;>")
print(ReceivedString)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Resize for easier calculation
    height, width, channels = frame.shape
    
    frame2 = imutils.resize(frame, width=700)
    (H, W) = frame2.shape[:2]

    ## Use blur detection to determine focus
    t_contrast = calculate_focus_score(frame2)

    half_w = 200
    half_h = 200

    # Get a center crop before drawing rectangle
    cent_crop = frame[int(height/2-half_h):int(height/2+half_h), int(width/2-half_w):int(width/2+half_w)] 
    cv2.imshow('crop', cent_crop)
    cent_contrast = calculate_focus_score(cent_crop)    
    
    if r is not None: # Track a certain bounding box and measure its focus:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame2)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]

            cropped_image = frame2[int(r[1]):int(r[1]+r[3]),  
                            int(r[0]):int(r[0]+r[2])] 
            c_contrast = calculate_focus_score(cropped_image)

            cv2.rectangle(frame2, (x, y), (x + w, y + h),
				(0, 255, 0), 2)
            
            
            # Average out the values for less noise
            if data_timer_f >= 10:
                # calculate averages and display
                crop_val = c_cont_sum/data_timer_f
                total_val = t_cont_sum/data_timer_f
                print("Crop: " + str(crop_val) + " vs. Crop Max: " + str(crop_max) + " vs. Total: " + str(total_val))
                data_timer_f = 0
                c_cont_sum = 0
                t_cont_sum = 0
                cent_cont_sum = 0
                # Microscope coarse-fine adjustments
                if focus_mode:
                    if s_coarse:
                        if crop_val < crop_max:
                            write_only("<R 100;>")
                        elif crop_val > crop_max and crop_val < (crop_max + crop_max * 1.2):
                            crop_max = crop_val - (crop_val * 0.2)
                            s_coarse = False
                            s_fine = True
                    elif s_fine:
                        if crop_val < crop_max:
                            write_only("<E 10;>")
                else: # reset if not in focus mode
                    crop_max = 150
                    s_coarse = True
                    s_fine = False
            else: # per loop incrementation
                total_collect[data_timer_f] = t_contrast
                data_timer_f += 1
                c_cont_sum += c_contrast
                t_cont_sum += t_contrast
            
        if not success: # reset all essential variables
            tracker = cv2.TrackerKCF_create()
            r = None
            crop_max = 150
            data_timer_f = 0
            c_cont_sum = 0
            t_cont_sum = 0
            cent_cont_sum = 0
            s_coarse = True
            s_fine = False
            write_only("<P 625;>")
    else: # To calculate focus of the central area and total frame
        # Draw center area rectangle - get frame copies before this line!
        cv2.rectangle(frame, (int(width/2-half_w), int(height/2-half_h)), (int(width/2+half_w), int(height/2+half_h)), (0, 255, 0), 2)
        # calculate averages and display
        if data_timer >= 10: 
            crop_val = cent_cont_sum/data_timer
            total_val = t_cont_sum/data_timer
            print("Center Crop: " + str(crop_val) + " vs Total: " + str(total_val))
            data_timer = 0
            t_cont_sum = 0
            cent_cont_sum = 0
            # Microscope coarse-fine adjustments
            if focus_mode:
                if s_coarse:
                    if crop_val < cent_max:
                        write_only("<R 100;>")
                    elif crop_val > cent_max and crop_val < (cent_max + cent_max * 1.2):
                        cent_max = crop_val - (crop_val * 0.2)
                        s_coarse = False
                        s_fine = True
                elif s_fine:
                    if crop_val < cent_max:
                        write_only("<E 10;>")
            else: # reset if not in focus mode
                cent_max = 15
                s_coarse = True
                s_fine = False
        else: # per loop incrementation
            total_collect[data_timer] = t_contrast
            data_timer += 1
            t_cont_sum += t_contrast
            cent_cont_sum += cent_contrast

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('frame2', frame2)

    # Poll for a key
    usr_key = cv2.waitKey(1)

    # Disable and enable focus mode
    if usr_key == ord('f'):
        if focus_mode:
            print("Focus mode disabled!")
            focus_mode = False
            write_only("<P 630;>")
        else:
            print("Focus mode enabled!")
            focus_mode = True

    # Zoom out/in
    if usr_key == ord('z'): # full zoom
        write_only("<Z 180;>")
    if usr_key == ord('x'): # home position
        write_only("<Z 0;>")
    if usr_key == ord('v'): # half zoom
        write_only("<Z 90;>")

    # Single out a frame to gauge a cropped area
    if usr_key == ord('p'):
        ## Manually select a region of interest (ROI)
        # Select ROI 
        r = cv2.selectROI("select the area", frame2) 

        # Will only give this if cancelled
        if r == (0, 0, 0, 0):
            print("Cancelling!")
            cv2.destroyAllWindows()
            r = None
            continue

        # Crop image 
        cropped_image = frame2[int(r[1]):int(r[1]+r[3]),  
                            int(r[0]):int(r[0]+r[2])] 
        
        cv2.imshow("Cropped image", cropped_image) 
        
        c_contrast = calculate_focus_score(cropped_image)
        print("Crop: " + str(c_contrast) + " vs. Total: " + str(t_contrast))

        tracker.init(frame2, r)

        # Wait for input to quit, clear windows
        if cv2.waitKey(0):
            cv2.destroyAllWindows()

    # Quit with q
    if usr_key == ord('q'):
        print("Quitting!")
        break
        
# When everything done, release the capture
SerialObj.close()
cap.release()
cv2.destroyAllWindows()

