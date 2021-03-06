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
images_and_descriptors = []
keypoints = create_keypoints(256, 256)

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.

for image in image_list:
    kp, des = sift.compute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), keypoints)
    # image = cv2.drawKeypoints(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    images_and_descriptors.append((image, des))
# YOUR CODE HERE

# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())

# YOUR CODE HERE
# load descriptors of input image
queue = PriorityQueue()
image_compare = cv2.imread('./images/db/query_car.jpg', 1)
# image_compare = cv2.imread('./images/db/query_face.jpg', 1)
# image_compare = cv2.imread('./images/db/query_flower.jpg', 1)
kp, desToCompare = sift.compute(cv2.cvtColor(image_compare, cv2.COLOR_BGR2GRAY), keypoints)

for imageData in images_and_descriptors:
    image, des = imageData
    queue.put((cv2.norm(desToCompare, des, cv2.NORM_L2), image))

# 5. output (save and/or display) the query results in the order of smallest distance
i = 0
numpy_horizontal = cv2.resize(image_compare, (200, 200))
while not queue.empty():
    (norm, image_queue) = queue.get()
    if i < 8:
        image_queue = cv2.resize(image_queue, (200, 200))
        numpy_horizontal = np.hstack((numpy_horizontal, image_queue))
        i += 1

cv2.imshow("Compare", numpy_horizontal)
cv2.waitKey()
