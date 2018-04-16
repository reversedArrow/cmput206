import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test.jpg',0)
#gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
print(img.shape)
kernel = -np.ones((3,3),np.float32)
kernel1 = -np.zeros((3,3),np.float32)
kernel2 = -np.zeros((3,3),np.float32)
kernel3 = np.ones((3,3),np.float32)/9
kernel[1][1] = 8;
kernel1[1][1] = 1;
kernel2[1][2] = 1;
dst = cv2.filter2D(img,-1,kernel)
dst1 = cv2.filter2D(img,-1,kernel1)
dst2 = cv2.filter2D(img,-1,kernel2)
dst3 = cv2.filter2D(img,-1,kernel3)
dst4 = img+(img-dst3)

		

plt.subplot(321),plt.imshow(img,'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(dst,'gray'),plt.title('Laplacian filter1')
plt.xticks([]), plt.yticks([])
plt.subplot(323),plt.imshow(dst1,'gray'),plt.title('Laplacian filter2')
plt.xticks([]), plt.yticks([])
plt.subplot(324),plt.imshow(dst2,'gray'),plt.title('Laplacian filter3')
plt.xticks([]), plt.yticks([])
plt.subplot(325),plt.imshow(dst3,'gray'),plt.title('Average filter')
plt.xticks([]), plt.yticks([])
plt.subplot(326),plt.imshow(dst4,'gray'),plt.title('convolution operation')
plt.xticks([]), plt.yticks([])
plt.show()