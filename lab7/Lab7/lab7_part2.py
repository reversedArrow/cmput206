import numpy as np
import cv2
import matplotlib.pyplot as plt
import watershed

def imreconstruct(marker, mask):
    curr_marker = np.copy(marker).astype(mask.dtype)
    kernel = np.ones([3, 3])
    while True:
        next_marker = cv2.dilate(curr_marker, kernel, iterations=1)
        intersection = next_marker > mask
        next_marker[intersection] = mask[intersection]
        if np.array_equal(next_marker, curr_marker):
            return curr_marker
        curr_marker = np.copy(next_marker)
    return curr_marker


def imimposemin(marker, mask):
    # adapted from its namesake in MATLAB
    fm = np.copy(mask)
    fm[marker] = -np.inf
    fm[np.invert(marker)] = np.inf
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        range = float(np.max(mask) - np.min(mask))
        if range == 0:
            h = 0.1
        else:
            h = range * 0.001
    else:
        # Add 1 to integer images.
        h = 1
    fp1 = mask + h
    g = np.minimum(fp1, fm)
    return np.invert(imreconstruct(
        np.invert(fm.astype(np.uint8)), np.invert(g.astype(np.uint8))
    ).astype(np.uint8))

sigma = 2.5
img_name = 'lab7.bmp'
img_rgb = cv2.imread(img_name).astype(np.float32)
img_gs = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

img_blurred = cv2.GaussianBlur(img_gs, (int(2 * round(3 * sigma) + 1), int(2 * round(3 * sigma) + 1)), sigma,
                     borderType=cv2.BORDER_REPLICATE)

[img_grad_y, img_grad_x] = np.gradient(img_blurred)
img_grad = np.square(img_grad_x) + np.square(img_grad_y)

# refined blob locations generated generated in part 3 of lab 6
blob_markers = np.loadtxt('blob_markers.txt', dtype=np.bool, delimiter='\t')

img_grad_min_imposed = imimposemin(blob_markers, img_grad)

markers = watershed.getRegionalMinima(img_grad_min_imposed)
plt.figure(0)
plt.imshow(markers)
plt.title('markers')

labels = watershed.iterativeMinFollowing(img_grad_min_imposed, markers)
plt.figure(1)
plt.imshow(labels)
plt.title('labels')

plt.show()

# add your code here
img,contours,hierarchy= cv2.findContours(labels,cv2.RETR_FLOODFILL ,cv2.CHAIN_APPROX_NONE)

#print(contours)

for i in range(len(contours)):
    area =  cv2.contourArea(contours[i])
    if area>10 and area<50:
        cv2.drawContours(img_gs, contours, i, (255,0,0))
plt.imshow(img_gs)
plt.title("Pruned Contours")
plt.show()


#reference: https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html?highlight=drawcontours#drawcontours
#refernece: https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#contourarea
#reference: https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours