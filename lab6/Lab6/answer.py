import cv2, math
from matplotlib import pyplot as plt
import numpy as np
import copy
import collections
import scipy.ndimage


class image:
    def __init__(self, filename):
        self.image = cv2.imread(filename, 0)
        self.filtered = []
        self.height = 0
        self.width = 0
        self.getdimension()

    def getdimension(self):
        dimension = self.image.shape
        self.height = dimension[0]
        self.width = dimension[1]
        return

    def gaussianBlur(self, sigma):
        k = int(2*round(3*sigma)+1)
        self.filtered = cv2.GaussianBlur(self.image, (k, k), sigma)

    # Create a Laplacian-of-Gaussian Volume
    def LoG(self, sigma):
        k = int(2*round(3*sigma)+1)
        self.LoGed = cv2.GaussianBlur(self.filtered, (k, k), sigma)
        self.LoGed = cv2.Laplacian(self.LoGed, ddepth=cv2.CV_64F, ksize=k)

    def blob(self, LoG):
        # Detect local minima
        localmin = scipy.ndimage.filters.minimum_filter(LoG, size=8, mode='reflect', cval=0.0, origin=0)
        # Convert local min values to binary mask
        self.mask = (LoG == localmin)
        self.mask = np.sum(self.mask, axis=2)
        x, y = np.nonzero(self.mask)
        plt.scatter(y, x, c='red')
        displayVertical({'Rough blobs detected in image': self.image})

    # Refined blob detection
    def otsublob(self):
        # http://docs.opencv.org/3.2.0/d7/d4d/tutorial_py_thresholding.html
        # Otsu's thresholding after Gaussian filtering
        ret1, th1 = cv2.threshold(self.filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for i in range(self.height):
            for j in range(self.width):
                if th1[i][j] < ret1:
                    self.mask[i][j] = 0

        x, y = np.nonzero(self.mask)
        plt.scatter(y, x, c='red')
        displayVertical({'Refined blobs detected in image': self.image})


# Put display list as a dictionary {"Title": image}
def displayVertical(imageList):
    keys = imageList.keys()
    cmd = int(len(imageList)*10+1)*10+1
    for i in range(len(imageList)):
        plt.subplot(cmd), plt.imshow(imageList[keys[i]]), plt.title(keys[i])
        cmd += 1
    plt.show()


def main():
    img = image("lab6.bmp")
    img.gaussianBlur(2)
    displayVertical({"Input Image": img.image, "Blurred Image": img.filtered})

    # 3 level of LoG
    img.LoG(3)
    # Making a deep copy!
    level1 = copy.copy(img.LoGed)
    img.LoG(4)
    level2 = copy.copy(img.LoGed)
    img.LoG(5)
    level3 = copy.copy(img.LoGed)
    displayVertical(collections.OrderedDict([("Level 1", level1), ("Level 2", level2), ("level 3", level3)]))

    # Part 2. Obtain a rough estimate of blob locations
    LoG = np.zeros((img.height, img.width, 3), np.float32)
    LoG[:, :, 0] = level1
    LoG[:, :, 1] = level2
    LoG[:, :, 2] = level3
    img.blob(LoG)

    # Part 3
    img.otsublob()

if __name__ == "__main__":
    main()