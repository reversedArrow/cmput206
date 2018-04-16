import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

img1 = cv2.imread('day.jpg',0)
img2 = cv2.imread('night.jpg',0)

hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
hist2 = cv2.calcHist([img2],[0],None,[256],[0,256])

plt.subplot(221),plt.imshow(img1,'gray')
plt.subplot(222),plt.plot(hist1)
plt.subplot(223),plt.imshow(img2,'gray')
plt.subplot(224),plt.plot(hist2)


plt.xlim([0,256])
plt.show()

result = cv2.compareHist(hist1,hist2,0)
print(" ")
print("compare histograms: ")
print(result)
print(" ")

height1 = np.size(img1,0)
width1 = np.size(img1,1)
height2 = np.size(img2,0)
width2 = np.size(img2,1)

coefficient1 = height1 * width1
coefficient2 = height2 * width2


bhattacharyya = 0

for i in range(0,256):
	bhattacharyya += math.sqrt((hist1[i]/coefficient1 * hist2[i]/coefficient2)) 
		
print("bhattacharyya coefficient is: ")
print(bhattacharyya)	
	
