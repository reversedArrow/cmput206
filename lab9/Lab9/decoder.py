# Jpeg decoding

import cv2
import numpy as np

# import zigzag functions
from zigzag import *

##################### step 5 #####################
# load h, w, block_size and padded_img from txt files

# load 'encoded.txt' into padded_img matrix.
# You should use np.loadtxt if you have already used np.savetxt to save them.
padded_img = np.loadtxt('encoded.txt')##### your code #####


# load 'size.txt' to get [h, w, block_size]
# You should use np.loadtxt if you have already used np.savetxt to save them.
[h, w, block_size] = np.loadtxt('size.txt')
##### your code #####

##################### step 6 #####################
# get the size of padded_img
[H, W] = padded_img.shape##### your code #####

# compute number of blocks by diving height and width of image by block size
# copy from step 1
# number of blocks in height
nbh = np.int(np.ceil(h/np.int(block_size)))##### your code #####

# number of blocks in width
nbw = np.int(np.ceil(w/np.int(block_size)))
##### your code #####

##################### step 7 #####################
# start decoding:
# divide encoded image into block size by block size (here: 8-by-8) blocks
# reshape it to one dimensional array (here: 64)
# use inverse zig-zag to reorder the array into a block
# apply 2D inverse discrete cosine transform


# iterate over blocks
for i in range(nbh):
    
        # Compute start row index of the block, same as encoder
        row_ind_1 = i * np.int(block_size)
        ##### your code #####
        
        # Compute end row index of the block, same as encoder
        row_ind_2 = row_ind_1 + np.int(block_size)
        ##### your code #####
        
        for j in range(nbw):
            
            # Compute start column index of the block, same as encoder
            col_ind_1 = j * np.int(block_size)
            ##### your code #####
            
            # Compute end column index of the block, same as encoder
            col_ind_2 = col_ind_1 + np.int(block_size)
            ##### your code #####
            
            # select the current block we want to process using calculated indices
            block = padded_img[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]
            
            # reshape the 2D block (here: 8-by-8) to one dimensional array (here: 64)
            reshaped= np.reshape(block, np.int(block_size) * np.int(block_size))##### your code #####
            
            # use inverse_zigzag function to scan and reorder the array into a block
            reordered = inverse_zigzag(reshaped, np.int(block_size), np.int(block_size))##### your code #####
            
            # apply 2D inverse discrete cosine transform to the reordered matrix
            IDCT = cv2.idct(reordered)##### your code #####
            
            # copy IDCT matrix into padded_img on current block corresponding indices
            ##### your code #####
            IDCT = np.reshape(IDCT, (np.int(block_size), np.int(block_size)))
            padded_img[row_ind_1: row_ind_2, col_ind_1: col_ind_2] = np.copy(IDCT)


padded_img = np.uint8(padded_img)
cv2.imshow('decoded padded image', padded_img)

##################### step 8 #####################
# get the original size (h by w) image from padded_img

decoded_img = padded_img[0: np.int(h), 0: np.int(w)]##### your code #####

cv2.imshow('decoded image', decoded_img)

##################################################

cv2.waitKey(0)
cv2.destroyAllWindows()




