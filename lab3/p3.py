import cv2
import numpy as np
import math

sigma = 50
window_name = 'Part I'

img = cv2.imread('ex1.jpg', 0).astype(np.uint8)
current_img = np.copy(img)

img_height = img.shape[0]
img_width = img.shape[1]

kernel_width = 2 * img_width + 1
kernel_height = 2 * img_height + 1

gaussian_kernel = np.zeros((kernel_height, kernel_width))
# cauchy_kernel = np.zeros((kernel_height, kernel_width))

def mouseHandler(event, mouse_x, mouse_y, flags=None, param=None):
    global current_img

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    print 'mouse_x =', mouse_x
    print 'mouse_y =', mouse_y

    blurred_img = cv2.GaussianBlur(current_img, (5, 5), 3).astype(np.float32)

    # compute the x and y extents of the mask to extract
    x1 = img_width - mouse_x
    y1 = img_height - mouse_y
    x2 = x1 + img_width
    y2 = y1 + img_height

    # extract the translated mask from the kernel
    mask = gaussian_kernel[y1:y2, x1:x2]
    # mask = cauchy_kernel[y1:y2, x1:x2]

    img1 = np.multiply(blurred_img, mask)
    img2 = np.multiply(current_img, 1-mask)
    current_img = cv2.add(img1, img2).astype(np.uint8)
    cv2.imshow(window_name, current_img)

def initGaussianKernel():
    print 'Initializing Gaussian Mask...'
    for x in xrange(kernel_width):
        for y in xrange(kernel_height):
            diff_x = float(img_width - x)
            diff_y = float(img_height - y)
            exp_factor = (diff_x * diff_x + diff_y * diff_y) / (sigma * sigma)
            gaussian_kernel[y, x] = math.exp(-exp_factor)


initGaussianKernel()
# initCauchyKernel()

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouseHandler)
cv2.imshow(window_name, current_img)
while True:
    if cv2.waitKey(1) == 27:
        break