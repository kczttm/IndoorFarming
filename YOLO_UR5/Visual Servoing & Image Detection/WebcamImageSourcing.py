import cv2

cam = cv2.VideoCapture(0)

counter = 0

while True:

    result, image = cam.read()

    if counter == 3:
        if result:
        
            # showing result, it take frame name and image 
            # output
            cv2.imshow("Test", image)
        
            # saving image in local storage
            cv2.imwrite("Test.png", image)

            # If keyboard interrupt occurs, destroy image 
            # window
            cv2.waitKey(0)
            cv2.destroyWindow("Test Image from USB Webcam")
            break
    
    counter += 1
# If captured image is corrupted, moving to else part
else:
    print("No image detected. Please! try again")
        
 

    

