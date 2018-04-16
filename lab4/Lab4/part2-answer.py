
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


def normalize(array):
    height, width = array.shape
    for r in range(0, height):
        for c in range(0, width):
            if array[r][c] > 255:
                array[r][c] = 255
            elif array[r][c] < 0:
                array[r][c]=0

    return array

# part I

img = cv2.imread('PeppersBayerGray.bmp', 0)

h, w = img.shape

# our final image will be a 3 dimensional image with 3 channels
rgb = np.zeros((h,w,3),np.uint8)


# reconstruction of the green channel IG

IG = np.copy(img) # copy the image into each channel

for row in range(0, h, 4): # loop step is 4 since our mask size is 4.
    for col in range(0, w, 4): # loop step is 4 since our mask size is 4.

        IG[row, col+1] = (int(img[row, col])+int(img[row, col+2]))/2
        IG[row, col+3] = (int(img[row, col + 2]) + int(img[row + 1, col + 3])) / 2

        IG[row+1, col] = (int(img[row, col]) + int(img[row+2, col])) / 2
        IG[row+1, col+2] = (int(img[row+1, col+1]) + int(img[row+1, col+3])+int(img[row, col+2])+int(img[row+2, col+2]))/4

        IG[row+2, col+1] = (int(img[row+2, col]) + int(img[row+2, col+2])+int(img[row+1, col+1])+int(img[row+3, col+1]))/4
        IG[row+2, col+3] = (int(img[row+1, col+3]) + int(img[row+3, col+3])) / 2

        IG[row+3, col] = (int(img[row+2, col]) + int(img[row+3, col+1])) / 2
        IG[row+3, col+2] = (int(img[row + 3, col + 1]) + int(img[row + 3, col + 3])) / 2

# reconstruction of the red channel IR

IR = np.copy(img)

for row in range(0, h, 4):
    for col in range(0, w, 4):
        IR[row, col+2] = (int(img[row, col+1])+int(img[row, col+3]))/2
        IR[row + 1, col + 1] = (int(img[row, col+1])+int(img[row+2, col+1]))/2
        IR[row + 1, col + 2] = (int(img[row, col+1])+int(img[row+2, col+1])+int(img[row, col + 3]) + int(img[row + 2, col + 3]))/4
        IR[row + 1, col + 3] = (int(img[row, col + 3]) + int(img[row + 2, col + 3])) / 2
        IR[row + 2, col + 2] = (int(img[row + 2, col+1])+int(img[row + 2, col + 3]))/2

        IR[row, col] = IR[row, col+1]
        IR[row + 1, col] = IR[row + 1, col + 1]
        IR[row + 2, col] = IR[row + 2, col + 1]

        IR[row + 3, col + 1] = IR[row + 2, col + 1]
        IR[row + 3, col + 2] = IR[row + 2, col + 2]
        IR[row + 3, col + 3] = IR[row + 2, col + 3]
        IR[row + 3, col] = IR[row + 3, col + 1]

# reconstruction of the blue channel IB

IB = np.copy(img)

for row in range(0, h, 4):
    for col in range(0, w, 4):
        IB[row + 1, col + 1] = (int(img[row + 1, col])+int(img[row + 1, col + 2]))/2
        IB[row + 2, col] = (int(img[row + 1, col])+int(img[row + 3, col]))/2
        IB[row + 2, col + 1] = (int(img[row + 1, col])+int(img[row + 3, col]) + int(img[row + 1, col + 2]) + int(img[row + 3, col + 2]))/4
        IB[row + 2, col + 2] = (int(img[row + 1, col + 2]) + int(img[row + 3, col + 2])) / 2
        IB[row + 3, col + 1] = (int(img[row + 3, col])+int(img[row + 3, col + 2]))/2

        IB[row, col] = IB[row + 1, col]
        IB[row, col + 1] = IB[row + 1, col + 1]
        IB[row, col + 2] = IB[row + 1, col + 2]
        IB[row, col + 3] = IB[row, col + 2]

        IB[row + 1, col + 3] = IB[row + 1, col + 2]
        IB[row + 2, col + 3] = IB[row + 2, col + 2]
        IB[row + 3, col + 3] = IB[row + 3, col + 2]


# merge the channels
# Red Channel
rgb[:, :, 0] = IR
# Green Channel
rgb[:, :, 1] = IG
# Blue Channel
rgb[:, :, 2] = IB


cv2.imwrite('rgb.jpg', rgb)

plt.imshow(rgb), plt.title('rgb')
plt.show()

# part II should be written here:

# Convert channel type to float
IR = IR.astype(np.float32)
IG = IG.astype(np.float32)
IB = IB.astype(np.float32)

DR = IR - IG
DB = IB - IG

# Apply 3x3 median filter
MR = cv2.medianBlur(DR, 3)
MB = cv2.medianBlur(DB, 3)

IRR = MR + IG
IBB = MB + IG

# Clip out out-of-bound values
IRR = normalize(IRR)
IBB = normalize(IBB)

# merge the channels
# Red Channel
rgb[:, :, 0] = IRR
# Green Channel
rgb[:, :, 1] = IG
# Blue Channel
rgb[:, :, 2] = IBB


cv2.imwrite('rgb.jpg', rgb)

plt.imshow(rgb), plt.title('advanced rgb')
plt.show()

cv2.destroyAllWindows()
