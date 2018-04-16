import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img = cv2.imread('test.jpg',0)
kernel = -np.ones((3,3),np.float32)

height,width = kernel.shape
#print(height,width)
pad = (width-1)/2

pad = int(pad)
#print(pad)

img1 = cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
h,w = img1.shape
print(h,w)
newimg = img.copy()
for i in range(0,height+1):
	for j in range(0,width+1):
		newimg[i][j] = np.sum(img1[i:i+height,j:j+width]*kernel)
		
plt.subplot(121),plt.imshow(img,'gray')
plt.subplot(122),plt.imshow(newimg,'gray')
plt.show()


#reference https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
#reference https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html