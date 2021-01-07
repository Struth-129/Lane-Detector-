import numpy as np
import cv2

def imglin(image,lines):
    lines_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0),3)
    images_lines = cv2.addWeighted(image,0.8,lines_image,1,0.0)
    return images_lines


def crop_fn(canny_image,region): 
    mask = np.zeros_like(canny_image)
    cv2.fillPoly(mask,region,255)
    masked = cv2.bitwise_and(canny_image,mask)
    return masked    

def detect(grey,frame):
    frame_blur = cv2.GaussianBlur(frame,(5,5),0)
    frame_canny = cv2.Canny(frame,80,130)
    height , width = (frame.shape[0],frame.shape[1])
    Pov_vehicle = [
        (0,height),
        (width/2,height*0.6),
        (width,height)
    ]
    crop = crop_fn(frame_canny,np.array([Pov_vehicle],np.int32))
    lines = cv2.HoughLinesP(crop, rho=2 , theta =np.pi/180,threshold=50,minLineLength=40,maxLineGap=150)
    result = imglin(frame,lines)
    return result
