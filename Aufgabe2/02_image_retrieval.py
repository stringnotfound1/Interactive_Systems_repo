import cv2
import glob
import numpy as np
from queue import PriorityQueue


############################################################
#
#              Simple Image Retrieval
#
############################################################


# implement distance function
def distance(a, b):
    # YOUR CODE HERE
    pass


def create_keypoints(w, h):
    keypointSize = 11
    keypoints_local = []

    for x in np.arange(keypointSize, h - keypointSize, keypointSize*2):
        for y in np.arange(keypointSize, w - keypointSize, keypointSize*2):
            keypoint_local = cv2.KeyPoint(x, y, keypointSize)
            keypoints_local.append(keypoint_local)

    return keypoints_local

# 1. preprocessing and load
images = glob.glob('./images/db/*/*.jpg')
image_list = []
for file_name in images:
    image_list.append(cv2.imread(file_name, 1))

sift = cv2.xfeatures2d.SIFT_create()

# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
descriptors = []
keypoints = create_keypoints(256, 256)

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.
i = 0
for image in image_list:
    kp, des = sift.compute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), keypoints)
    image = cv2.drawKeypoints(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), keypoints, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    descriptors.append(des)
cv2.imshow("Test", image_list[9])
cv2.waitKey(0)
# YOUR CODE HERE

# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())

# YOUR CODE HERE
# image_list[10]
# load descriptors of input image
kp, des = sift.compute(cv2.cvtColor(image_list[10], cv2.COLOR_BGR2GRAY), keypoints)

# 5. output (save and/or display) the query results in the order of smallest distance

# YOUR CODE HERE
