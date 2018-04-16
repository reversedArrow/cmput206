import numpy as np
import cv2
import scipy.ndimage
from matplotlib import pyplot as plt
import random

#1 Read this image and convert it to gray scale. and get height and width for later use.
img = cv2.imread('lab6.bmp', 0).astype(np.float64)
h,w = img.shape

#2 Apply Gaussian filtering using  cv2.GaussianBlur with a suitably chosen sigma and the kernel size k computed as 

sigma = 2

k = int(2*round(3*sigma)+1)

blur = cv2.GaussianBlur(img, (k,k), sigma)

#3 Show both the original image and its blurred version together in the same figure

plt.subplot(211), plt.imshow(img), plt.title('Input Image')
plt.subplot(212), plt.imshow(blur), plt.title('Blurred Image')
plt.tight_layout()
plt.show()



#4 Create a 3 level Laplacian-of-Gaussian (LoG) volume by applying this filter at 3 different scales or sigma values to the blurred image obtained in step 2

level=[]
sigma_value = [3,4,5]
counter = [0,1,2]
old_img = np.copy(blur)


# 4.3 The 3 sigma values should be chosen to give best results; using consecutive integers such as 3, 4 and 5 might be a good starting point
for i in sigma_value:
    k = int(2*round(3*i)+1) # 4.2 The kernel size should be computed using the same expression as in step 2
    # 4.1 The LoG filter can be applied using cv2.GaussianBlur followed by cv2.Laplacian
    blur = cv2.GaussianBlur(old_img, (k,k), i)
    log = cv2.Laplacian(blur,-1,ksize = k)   
    level.append(log) #append in index for later use.
    for j in counter:
        new = np.zeros((h,w,3),np.float64) # 4.4 All 3 levels of the volume must be stored in a single  Numpy array where h and w are the height and width of the input image    

        #use built_in log
        new[ :, :, j] = log
        
        log = cv2.Laplacian(log,-1,ksize = k)


#5 Display the 3 levels of the volume together in the same figure         
plt.suptitle('LoG Pyramid', size = 14)
plt.subplot(311), plt.imshow(level[0]), plt.title('Level 1')
plt.subplot(312), plt.imshow(level[1]), plt.title('Level 2')
plt.subplot(313), plt.imshow(level[2]), plt.title('Level 3')
        
plt.tight_layout()
plt.subplots_adjust(top = 0.9)
plt.show()       


imgsize = 4
min_filter_size = 20
# OpenCV does not provide a function to perform this so you can either use the scipy function scipy.ndimage.filters.minimum_filter
#binary = (scipy.ndimage.filters.minimum_filter(new, imgsize)==new)
lm = scipy.ndimage.filters.minimum_filter(new, min_filter_size)

# It should also be noted that the scipy function returns the actual values of the detected minima rather than their locations so additional steps will be needed to convert its output to the required  binary image.
msk = (new == lm)
# Collapse this 3D binary image into a single channel image by computing the sum of corresponding pixels in the 3 channels. This can be done using np.sum
imgsum = np.sum(msk,axis=2)
# Show the locations of all non zero entries in this collapsed array overlaid on the input image as red points
pointimg = np.nonzero(imgsum)


plt.scatter(pointimg[1],pointimg[0],edgecolors = 'none', c = 'r')
plt.imshow(img)
plt.gca().invert_yaxis()
plt.show()


#part3


#Apply Otsu thresholding on the blurred image computed in step 2 of part 1 using  cv2.threshold to obtain the optimal threshold for this image
#Remove all minima detected in part 2 where the pixel values in this image are less than this threshold 
t, J_otsu = cv2.threshold(blur.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imgsum = np.multiply(imgsum, blur >= t)

I_rgb = cv2.imread('lab6.bmp').astype(np.float32)
I = cv2.cvtColor(I_rgb, cv2.COLOR_BGR2GRAY)
levelnu = 0
[y, x] = imgsum.nonzero()
level.append(plt.figure(levelnu))
levelnu += 1




plt.imshow(I)
plt.scatter(x, y, edgecolors = 'none', c = 'r')
plt.xlim([0, I.shape[1]]),plt.ylim([0, I.shape[0]])
plt.title('blobs detection')

plt.show()

#reference:https://docs.opencv.org/3.0-beta/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#threshold
#reference:https://docs.opencv.org/3.2.0/d7/d4d/tutorial_py_thresholding.html