import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img = cv2.imread('test.jpg',0)


histogram = [0]*256
for i in range(0,182):
	for j in range (0,256):
		histogram[img[i][j]]+=1
		

plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.plot(histogram)
plt.xlim([0,256])

plt.show()

