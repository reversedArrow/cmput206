import cv2
import numpy as np
from matplotlib import pyplot as plt


#I don't know why but it's going to take a lot of time to processing 'for i in range(size)' in line 23. but it will work after it loads up. I change it to 1000 to get a better demo.

img = cv2.imread('damaged_cameraman.bmp')
msk = cv2.imread('damage_mask.bmp')
newimg = cv2.imread('damaged_cameraman.bmp')

height = np.size(newimg,0)
width = np.size(newimg,1)
size = np.size(img)
print(height,width,size)
damagePixel = []

#print damagedpixel in height and width
for i in range(height):
    for j in range(width):
        if msk[i][j].all() == 0:
            damagePixel.append((i,j))


#apply gaussian filter and merge into damaged picture.          
for i in range(1000):
    gaussian = cv2.GaussianBlur(newimg,(5,5),0)
    for j in damagePixel:
        newimg[j] = gaussian[j]
        
        
        
plt.subplot(121),plt.imshow(img),plt.title('Damaged Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(newimg),plt.title('After Inpainting')
plt.xticks([]), plt.yticks([])

plt.show()