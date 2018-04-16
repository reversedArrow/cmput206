import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.jpg',0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])

hist,bins = np.histogram(img.ravel(),256,[0,256])


plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.plot(hist)
plt.xlim([0,256])

plt.show()

