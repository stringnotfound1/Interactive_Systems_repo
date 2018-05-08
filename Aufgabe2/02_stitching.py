import numpy as np
import cv2
import math
import sys

# from ImageStitcher import *
from Aufgabe2.ImageStitcher import *

############################################################
#
#                   Image Stitching
#
############################################################

# 1. load panorama images
pano1 = cv2.imread('images/pano1.jpg')
pano2 = cv2.imread('images/pano2.jpg')
pano3 = cv2.imread('images/pano3.jpg')
pano4 = cv2.imread('images/pano4.jpg')
pano5 = cv2.imread('images/pano5.jpg')
pano6 = cv2.imread('images/pano6.jpg')

# print(pano1)
# order of input images is important is important (from right to left)
# cv2.imshow("Abc", pano1)
# cv2.waitKey(0)
# imageStitcher = ImageStitcher([pano3, pano2, pano1])   # list of images
imageStitcher = ImageStitcher([pano6, pano5, pano4])   # list of images
(matchlist, result) = imageStitcher.stitch_to_panorama()

if not matchlist:
    print("We have not enough matching keypoints to create a panorama")
else:

    print("We have matching keypoints to create a panorama")
    # YOUR CODE HERE
    # output all matching images
    # output result
    # Note: if necessary resize the image
    # cv2.imshow("test", matchlist[0])
    numpy_vertical = matchlist[0]
    for image in matchlist:
        numpy_horizontal = np.hstack((numpy_vertical, image))
    cv2.imshow("test", result)
    cv2.imshow("numpy_vertical", numpy_horizontal)
    cv2.waitKey(0)

