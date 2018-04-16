import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ex2.jpg',0)
edges = cv2.Canny(img,100,200)

kernel1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel2 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])


horizontal = cv2.filter2D(img,-1,kernel1)
vertical = cv2.filter2D(img,-1,kernel2)

gradient = horizontal + vertical


plt.subplot(321),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(323),plt.imshow(horizontal,cmap = 'gray')
plt.title('horizontal'),plt.xticks([]),plt.yticks([])
plt.subplot(324),plt.imshow(vertical,cmap = 'gray')
plt.title('vertical'),plt.xticks([]),plt.yticks([])
plt.subplot(325),plt.imshow(gradient,cmap = 'gray')
plt.title('gradient'),plt.xticks([]),plt.yticks([])


plt.show()