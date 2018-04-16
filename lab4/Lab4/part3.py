import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


# part I

img = cv2.imread('PeppersBayerGray.bmp', 0)

h,w = img.shape

# our final image will be a 3 dimentional image with 3 channels
rgb = np.zeros((h,w,3),np.uint8);


IG = np.copy(img) # copy the image into each channel
IG = IG.astype(np.float32)

for row in range(0,h,4):
    for col in range(0,w,4): 
        
        IG[row,col+1]=(int(img[row,col])+int(img[row,col+2]))/2         #B
        IG[row+1,col]=(int(img[row,col])+int(img[row+2,col]))/2         #E
        IG[row+3,col]=(int(img[row+2,col])+int(img[row+3,col+1]))/2     #M
        IG[row,col+3]=(int(img[row,col+2])+int(img[row+1,col+3]))/2     #D
        IG[row+1,col+2]=(int(img[row,col+2])+int(img[row+1,col+1])+int(img[row+1,col+3])+int(img[row+2,col+2]))/4  #G
        IG[row+2,col+1]=(int(img[row+2,col])+int(img[row+1,col+1])+int(img[row+2,col+2])+int(img[row+3,col+1]))/4  #J
        IG[row+2,col+3]=(int(img[row+1,col+3])+int(img[row+3,col+3]))/2 #L
        IG[row+3,col+2]=(int(img[row+3,col+1])+int(img[row+3,col+3]))/2 #O


IR = np.copy(img) # copy the image into each channel
IR = IR.astype(np.float32)

for row in range(0,h,4): 
    for col in range(0,w,4): 
        IR[row+1,col+1]=(int(img[row,col+1])+int(img[row+2,col+1]))/2   #F
        IR[row,col+2]=(int(img[row,col+1])+int(img[row,col+3]))/2       #C
        IR[row+1,col+3]=(int(img[row,col+3])+int(img[row+2,col+3]))/2   #H
        IR[row+2,col+2]=(int(img[row+2,col+3])+int(img[row+2,col+1]))/2 #K
        IR[row+1,col+2]=(int(img[row,col+1])+int(img[row,col+3])+int(img[row+2,col+1])+int(img[row+2,col+3]))/4    #G
        IR[row,col]=IR[row,col+1]                                       #A
        IR[row+1,col]=IR[row+1,col+1]                                   #E
        IR[row+2,col]=IR[row+2,col+1]                                   #I
        IR[row+3,col+3]=IR[row+2,col+3]                                 #P
        IR[row+3,col+2]=IR[row+2,col+2]                                 #O
        IR[row+3,col+1]=IR[row+2,col+1]                                 #N
        IR[row+3,col]=IR[row+2,col+1]                                   #M

IB = np.copy(img) 
IB = IB.astype(np.float32)

for row in range(0,h,4): 
    for col in range(0,w,4): 
        
        IB[row+1,col+1]=(int(img[row+1,col])+int(img[row+1,col+2]))/2   #F
        IB[row+1,col+3]=IB[row+1,col+2]                                 #H
        IB[row,col]=IB[row+1,col]                                       #A
        IB[row,col+1]=IB[row+1,col+1]                                   #B
        IB[row,col+2]=IB[row+1,col+2]                                   #C
        IB[row,col+3]=IB[row+1,col+3]                                   #D
        IB[row+2,col]=(int(img[row+1,col])+int(img[row+3,col]))/2       #I
        IB[row+2,col+1]=(int(img[row+1,col])+int(img[row+1,col+2])+int(img[row+3,col])+int(img[row+3,col+2]))/4     #J
        IB[row+2,col+2]=(int(img[row+1,col+2])+int(img[row+3,col+2]))/2 #K
        IB[row+2,col+3]=IB[row+2,col+2]                                 #L
        IB[row+3,col+1]=(int(img[row+3,col])+int(img[row+3,col+2]))/2   #N
        IB[row+3,col+3]=IB[row+3,col+2]                                 #P


# merge the channels
rgb[:,:,0]=IR
rgb[:,:,1]=IG
rgb[:,:,2]=IB

cv2.imwrite('rgb.jpg',rgb);

plt.imshow(rgb),plt.title('rgb')
plt.show()

# part II should be written here:

DR = IR-IG
DB = IB-IG

# apply 3x3 median filter
MR = cv2.medianBlur(DR,3)
MB = cv2.medianBlur(DB,3)

IRR = MR + IG
IBB = MB + IG

#remove out of bound values
for i in range(h):
    for j in range(w):
        if IRR[i,j] > 255:
            IRR[i,j] = 255
        elif IRR[i,j] < 0:
            IRR[i,j] = 0        
        if IBB[i,j] > 255:
            IBB[i,j] = 255
        elif IBB[i,j] < 0:
            IBB[i,j] = 0


# merge the channels
rgb[:,:,0]=IRR
rgb[:,:,1]=IG
rgb[:,:,2]=IBB
cv2.imwrite('rgb.jpg',rgb)


plt.imshow(rgb),plt.title('advanced')
plt.show()