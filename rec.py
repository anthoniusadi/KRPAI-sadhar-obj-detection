import numpy as np
from time import sleep
import operator
import cv2
import sys, os
from time import sleep
#import retinex
cap = cv2.VideoCapture(2)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('simulasi1.mp4',fourcc,25.0,(640,480))

def rescale_frame(frame,percent=100):
    width=int(frame.shape[1]* percent/100)
    height=int(frame.shape[0]* percent/100)
    dim = (width,height)
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
while (cap.isOpened()):
    ret, frame = cap.read()
    if(ret==True):
            
        frame = cv2.flip(frame, 1) 
        out.write(frame)
        cv2.imshow("Original", frame)   
        interrupt = cv2.waitKey(10)
        # if interrupt & 0xFF == ord('c'):
        #     cv2.imwrite(directory+'img_original'+'.jpg',original)
        if interrupt & 0xFF == 27: # esc key
            
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            break

cap.release()
out.release()
cv2.destroyAllWindows()
