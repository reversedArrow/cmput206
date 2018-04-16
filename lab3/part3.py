import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import random

#[i for i in dir(cv2) if 'left-button down' in i]
events = [i for i in dir(cv2) if 'EVENT' in i]
print events

img = cv2.imread('ex2.jpg',0).astype(np.uint8)
newimg = np.copy(img)

height = np.size(img,0)
width = np.size(img,1)
print(height,width)
#gaussian blur kernel
kernel = -np.ones(((2*height+1),(2*width+1)),np.float32)
gaussian_img = cv2.GaussianBlur(newimg, (15,15), 0)

def draw_circle(event,cursor_x,cursor_y,flags,param):
    global newimg
    
    if event == cv2.EVENT_RBUTTONDOWN:
        newdelta = random.randint(50,100)
        delta = newdelta
        print(delta)    
        
    if event == cv2.EVENT_LBUTTONDOWN:
        #cv2.circle(newimg,(cursor_x,cursor_y),100,(255,0,0),-1)
        
        
        #calculate border for new mask, Extract a region from the mask
        y = height - cursor_y; x = width - cursor_x
        new_y = height + y; new_x = width  + x
        gaussian = kernel[y:new_y,x:new_x]
        
        # implement the weighted averaging operation
        img1 = np.multiply(gaussian_img,gaussian)
        img2 = np.multiply(newimg,1-gaussian)
        newimg = np.add(img1,img2).astype(np.uint8)
        cv2.imshow('part3', newimg)
        

#Computing Weights gaussian distribution
delta = random.randint(30,70)
print(delta)
kernel_h = 2*height+1
kernel_w = 2*width+1
for i in range(kernel_h):
    for j in range(kernel_w):
        kernel[i,j] = math.exp(-((j-width)*(j-width)+(i-height)*(i-height))/(delta*delta))
        
cv2.namedWindow('part3')
cv2.setMouseCallback('part3',draw_circle)        


while(1):
    cv2.imshow('part3',newimg)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()