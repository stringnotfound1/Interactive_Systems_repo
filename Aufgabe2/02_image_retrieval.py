import cv2
import glob
from Queue import PriorityQueue

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
    keypoints = []
    keypointSize = 11

    # YOUR CODE HERE

    return keypoints


# 1. preprocessing and load
images = glob.glob('./images/db/*/*.jpg')

# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
descriptors = []
keypoints = create_keypoints(256, 256)

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.

# YOUR CODE HERE

# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())

# YOUR CODE HERE

# 5. output (save and/or display) the query results in the order of smallest distance

# YOUR CODE HERE
