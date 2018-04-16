# Jpeg encoding

import cv2
import numpy as np

# import zigzag functions
from zigzag import *

# defining block size
block_size = 8

# reading image in grayscale
img = cv2.imread('pout.tif', 0)
cv2.imshow('input image', img)

# get size of the image
[h , w] = img.shape


##################### step 1 #####################
# compute number of blocks by diving height and width of image by block size

# you need to convert h and w to float to get the right number
h = np.float32(h)
w = np.float32(w)

# to cover the whole image the number of blocks should be ceiling of the division of image size by block size
# at the end convert it to int

# number of blocks in height
nbh = int(np.ceil(h / block_size))
print(nbh)

# number of blocks in width
nbw = int(np.ceil(w / block_size))
print(nbw)

##################### step 2 #####################
# Pad the image, because sometime image size is not dividable to block size
# get the size of padded image by multiplying block size by number of blocks in height/width

# height of padded image
H =  nbh * block_size

# width of padded image
W =  nbw * block_size

# create a numpy zero matrix with size of H,W
padded_img = np.zeros((H,W))

# copy the values of img  into padde_img[0:h,0:w]
nph = np.int(h)
npw = np.int(w)
print(nph)
print(npw)
for x in range(0,nph):
    for y in range(0,npw):
        padded_img[x][y] = img[x][y]

cv2.imshow('input padded image', np.uint8(padded_img))

##################### step 3 #####################
# start encoding:
# divide image into block size by block size (here: 8-by-8) blocks
# To each block apply 2D discrete cosine transform
# reorder DCT coefficients in zig-zag order
# reshaped it back to block size by block size (here: 8-by-8)


# iterate over blocks
for i in range(nbh):
    
        # Compute start row index of the block
        row_ind_1 = i * block_size
        
        # Compute end row index of the block
        row_ind_2 = row_ind_1 + block_size
        
        for j in range(nbw):
            
            # Compute start column index of the block
            # block size
            col_ind_1 = j * block_size
            
            # Compute end column index of the block
            # 
            col_ind_2 = col_ind_1 + block_size
            


            # select the current block we want to process using calculated indices
            block = padded_img[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]
            
            # apply 2D discrete cosine transform to the selected block
            # you should use opencv dct function
            DCT = cv2.dct(block)
            
            # reorder DCT coefficients in zig zag order by calling zigzag function
            # it will give you a one dimentional array
            reordered = zigzag(DCT)
            
            # reshape the reorderd array back to (block size by block size) (here: 8-by-8)
            reshaped = np.reshape(reordered, (block_size, block_size))
            
            # copy reshaped matrix into padded_img on current block corresponding indices
            padded_img[row_ind_1:row_ind_2 , col_ind_1:col_ind_2] = reshaped

cv2.imshow('encoded image', np.uint8(padded_img))

##################### step 4 #####################
# write h, w, block_size and padded_img into txt files at the end of encoding

# write padded_img into 'encoded.txt' file. You can use np.savetxt function.
np.savetxt('encoded.txt', padded_img)

# # write [h, w, block_size] into size.txt. You can use np.savetxt function.
np.savetxt('size.txt', [h, w, block_size])

##################################################

cv2.waitKey(0)
cv2.destroyAllWindows()




