import cv2
import numpy as np
from matplotlib import pyplot as plt



def nothing(x):
    pass

# Create a black image, a window
img = cv2.imread('ex1.jpg',0)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('H','image',0,255,nothing)
cv2.createTrackbar('W','image',0,255,nothing)


# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

while(1):
   
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    H = cv2.getTrackbarPos('H','image')
    W = cv2.getTrackbarPos('W','image')
 
    s = cv2.getTrackbarPos(switch,'image')
	
	

    if s == 0:
        edges = img

    else:
        edges = cv2.Canny(img,H,W)
    cv2.imshow('image',edges)
	

cv2.destroyAllWindows()
cv2.destroyAllWindows()