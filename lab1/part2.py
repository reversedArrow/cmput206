import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('test.jpg',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])


equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))
cv2.imshow('test2.jpg',res)

