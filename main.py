#Author : Saket Srivastava


import cv2
import numpy as np
from detect import detect

cap = cv2.VideoCapture(0) ##Capture video to process( 0 means your laptop camera)

##OpenCv loops starts
while True:
    ret, frame = cap.read() ##Storing second frames of video to frame variable
    if ret is False:  ##If Video Ended
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    result = detect(gray,frame)
    cv2.imshow("Detector",result)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()         