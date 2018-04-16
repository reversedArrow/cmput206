import numpy as np
import watershed

test_img = np.loadtxt('lab7_part1.txt', dtype=np.uint8, delimiter='\t')
print 'test_img:\n', test_img

test_markers = watershed.getRegionalMinima(test_img)
print 'test_markers:\n', test_markers


test_labels = watershed.iterativeMinFollowing(test_img, test_markers)
print 'test_labels:\n', test_labels

