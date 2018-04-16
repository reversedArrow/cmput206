import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test.jpg',0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])

height = np.size(img,0)
width = np.size(img,1)


accumulateHist = []
dis = 0

accumulateHist.append(0)
for i in range(1,256):
    dis = accumulateHist[i-1]
    accumulateHist.append(hist[i]+dis) #defined by accumulating histogram
    
	
	
equalizedImg = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        pt = img[i,j] #reference: I ask a friend about this algorithm
        equalizedImg[i,j] = int((255*accumulateHist[pt])/(height*width)+0.5) #function defined
        
equalizedHist = []
for i in range(256):
    equalizedHist.append(0)
for i in range(height):
    for j in range(width):
        equalizedHist[int(equalizedImg[i,j])] += 1

		

hist = cv2.calcHist([img],[0],None,[256],[0,256])


plt.subplot(221)
plt.imshow(img,'gray')
plt.subplot(222)
plt.plot(hist)
plt.subplot(223)
plt.imshow(equalizedImg, 'gray')
plt.subplot(224)
plt.plot(equalizedHist)

plt.show()