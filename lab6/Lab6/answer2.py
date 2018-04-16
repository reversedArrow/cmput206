import numpy as np
import cv2, scipy.ndimage
from matplotlib import pyplot as plt

img = cv2.imread('lab6.bmp', 0).astype(np.float64)
h,w = img.shape

sigma = 2.5
k = int(2*round(3*sigma)+1)
blur = cv2.GaussianBlur(img, (k,k), sigma)
#############################
##### Part I 1_3 Output #####

#plt.subplot(211), plt.imshow(img), plt.title('Input Image')
#plt.subplot(212), plt.imshow(blur), plt.title('Blurred Image')

#plt.tight_layout()
#plt.show()
##############################
grp = []
new = np.zeros((h,w,3),np.float64)

num = 0
sigmas = [3,4,5]
oldlog = img
for values in sigmas:
    k = int(2*round(3*values)+1)
    blur = cv2.GaussianBlur(oldlog, (k,k), values)
    log = cv2.Laplacian(blur,-1,ksize = k)
    grp.append(log)
    new[:,:,num] = log
    log = cv2.Laplacian(log,-1,ksize = k)
    oldlog = log
    num += 1

#############################
##### Part I 1_5 Output #####

#plt.suptitle('LoG Pyramid', size = 14)
#plt.subplot(311), plt.imshow(grp[0]), plt.title('Level 1')
#plt.subplot(312), plt.imshow(grp[1]), plt.title('Level 2')
#plt.subplot(313), plt.imshow(grp[2]), plt.title('Level 3')

#plt.tight_layout()
#plt.subplots_adjust(top = 0.9)
#plt.show()
##############################   
imgsize = 4
binary = (scipy.ndimage.filters.minimum_filter(new, imgsize)==new)
imgsum = np.sum(binary,axis=2)
pointimg = np.nonzero(imgsum)
############################## 
##### Part II 2_3 Output #####

plt.scatter(pointimg[1],pointimg[0],edgecolors = 'none', c = 'r')
plt.imshow(img)
plt.gca().invert_yaxis()
plt.show()
############################## 