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

labels = watershed.iterativeMinFollowing(img_grad_min_imposed, np.copy(markers))
plt.figure(1)
plt.imshow(labels)
plt.title('labels')






#3
img, contours, hierarchy = cv2.findContours(
    np.copy(labels), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)
hierarchy = hierarchy[0]
contour_id = 0
pruned_contours = []
n_pruned_contours = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if hierarchy[contour_id - 1][3] >= 0 or area >= 50 or area <= 10:
        pass
    else:
        pruned_contours.append(contour)
    contour_id += 1

img_contours = cv2.drawContours(img_gs, pruned_contours, -1, (255, 0, 0),  1)
plt.figure(2)
plt.imshow(img_contours)
plt.title('Pruned Contours')

plt.show()