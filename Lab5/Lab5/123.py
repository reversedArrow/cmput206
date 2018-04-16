import numpy as np
import cv2
from matplotlib import pyplot as plt


img1 = cv2.imread('im1.jpg', 0)
img2 = cv2.imread('im2.jpg',0)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]


# Initiate BRISK detector
# your code #
detector = cv2.BRISK_create()
# Find the keypoints and descriptors with BRISK
# your code #
key1,d1 = detector.detectAndCompute(img2,None)
key2,d2 = detector.detectAndCompute(img1,None)

len1 = len(key1)
len2 = len(key2)
print(len1,len2)

# initialize Brute-Force matcher
# your code #
matcher = cv2.BFMatcher()

# use KNN match of Brute-Force matcher for descriptorsm
# your code #


matches = matcher.knnMatch(d1,d2,k=2)
#match = cv2.drawMatchesKnn(img1,key1,img2,key2,matches[:10],flags = 2)
#cv2.imshow(match)


#exclude outliers
# your code #
# ratio test, keep the good (avaliable) points add them in a list.
good = []
for m,n in matches:
  if m.distance < 0.75*n.distance:
    good.append(m)
    
# Compute homography matrix M
# your code #
#FLANN_INDEX_KDTREE = 0
#index_points =  dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_points = dict(checks=50)
min_match = 10
if len(good)>min_match:
  src_pts = np.float32([ key1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
  dst_pts = np.float32([ key2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

  M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
  matchdone = mask.ravel().tolist()
else:
  print "Matches are not enough - %d/%d" % (len(good),min_match)
  matchdone = None

#if len(good)>min_match:
  
  #img3 = np.copy(img1)
  #img3 = cv2.drawMatches(img1,key1,img2,key2,good[:],img3,flags=2)

  #plt.imshow(img3),plt.show()


#if match is done (# your code #):


#if match is done (# your code #):
if matchdone:

  # Initialize a matrix to include all the coordinates in the image, from (0, 0), (1, 0), ..., to (w-1, h-1)
  # In this way, you do not need loops to access every pixel

  # Calculate the new image coordinates based on the homography matrix
  c = np.zeros((3, h2*w2), dtype=np.int)
  for y in range(h2):
    c[:, y*w2:(y+1)*w2] = np.matrix([np.arange(w2), [y] * w2,  [1] * w2])
  new_c = M * np.matrix(c)
  new_c = np.around(np.divide(new_c, new_c[2]))

  # The new coordinates may have negative values. So perform translation if necessary
  x_min = int(np.amin(new_c[0]))
  y_min = int(np.amin(new_c[1]))
  x_max = int(np.amax(new_c[0]))
  y_max = int(np.amax(new_c[1]))
  if x_min < 0:
    t_x = -x_min
  else:
    t_x = 0
  if y_min < 0:
    t_y = -y_min
  else:
    t_y = 0

  # Initialize the final image to include every pixel of the stitched images  
  new_w = int(np.maximum(x_max, w1) - np.minimum(x_min, 0) + 1)
  new_h = int(np.maximum(y_max, h1) - np.minimum(y_min, 0) + 1)

  new_img1 = np.zeros((new_h, new_w), dtype=np.uint8)
  new_img2 = np.zeros((new_h, new_w), dtype=np.uint8)

  # Assign the first image
  new_img1[t_y:t_y+h1, t_x:t_x+w1] = img1

  # Assign the second image based on the newly calculated coordinates
  for idx in range(c.shape[1]):
    x = c[0, idx]
    y = c[1, idx]
    x_c = int(new_c[0, idx])
    y_c = int(new_c[1, idx])
    new_img2[y_c + t_y, x_c + t_x] = img2[y, x]

  # The stitched image
  new_img = (new_img1 + new_img2) / 2
  cv2.imwrite('stitched_img.jpg', new_img);
  cv2.imshow("Stitched Image", new_img)
  cv2.waitKey()
  cv2.destroyAllWindows() 

#reference: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html