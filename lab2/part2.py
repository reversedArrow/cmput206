import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('noisy.jpg')

median = cv2.medianBlur(img, 5)
gaussian = cv2.GaussianBlur(img,(5,5),0)

plt.subplot(131),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(median),plt.title('Median Filtering')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(gaussian),plt.title('Gaussian Filtering')
plt.xticks([]), plt.yticks([])
plt.show()