import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img = cv2.imread('test.jpg',0)
#kernel = -np.ones((3,3),np.float32)

k_laplacian = np.array([[-1, -1, -1] ,[-1, 8, -1], [-1, -1, -1]],dtype= "int")

height,width = k_laplacian.shape
print(height,width)
pad = (width-1)/2

pad = int(pad)
print(pad)

img1 = cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
h,w = img1.shape

laplacian = cv2.filter2D(img,-1,k_laplacian)

plt.subplot(121),plt.imshow(img,'gray')
plt.subplot(122),plt.imshow(laplacian,'gray')
plt.show()


