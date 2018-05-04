import numpy as np
import cv2
import math
import sys
from ImageStitcher import *


############################################################
#
#                   Image Stitching
#
############################################################

# 1. load panorama images


# order of input images is important is important (from right to left)
imageStitcher = ImageStitcher([]) # list of images
(matchlist, result) = imageStitcher.stitch_to_panorama()

if not matchlist:
    print("We have not enough matching keypoints to create a panorama")
else:
    # YOUR CODE HERE
    # output all matching images
    # output result
    # Note: if necessary resize the image

