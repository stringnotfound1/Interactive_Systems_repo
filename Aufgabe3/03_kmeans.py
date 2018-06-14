import numpy as np
import cv2
import math
import sys


############################################################
#
#                       KMEANS
#
############################################################

# implement distance metric - e.g. squared distances between pixels
def distance(a, b):
    return cv2.norm(a, b, cv2.NORM_L2)


# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the error
def initialize(img):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""
    # YOUR CODE HERE

    for i in range(0, numclusters):
        rng_h = int(height * np.random.random())
        rng_w = int(width * np.random.random())
        current_cluster_centers[i] = img[rng_h][rng_w]


def assign_to_current_mean(img, result, clustermask):
    """The function expects the img, the resulting image and a clustermask.
    After each call the pixels in result should contain a cluster_color corresponding to the cluster
    it is assigned to. clustermask contains the cluster id (int [0...num_clusters]
    Return: the overall error (distance) for all pixels to there closest cluster center (mindistance px - cluster center).
    """
    overall_dist = 0

    for w in range(0, width):
        for h in range(0, height):
            lowestDistance = sys.maxsize
            for cluster_id, color in enumerate(current_cluster_centers):
                dist = distance(img[w, h], color[0])
                if dist < lowestDistance:
                    cluster_id_temp = cluster_id
                    lowestDistance = dist
                    clusterColor = color[0]
            clustermask[w, h] = cluster_id_temp
            overall_dist += lowestDistance
            result[w, h] = clusterColor
    return overall_dist


def update_mean(img, clustermask):
    """This function should compute the new cluster center, i.e. numcluster mean colors"""

    for cluster_id, color in enumerate(current_cluster_centers):
        values = []
        for h in range(0, height):
            for w in range(0, width):
                if clustermask[h, w] == cluster_id:
                    values.append(img[h, w])
        current_cluster_centers[cluster_id] = np.mean(values, axis=0)

    pass


def kmeans(img):
    """Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    # initializes each pixel to a cluster
    # iterate for a given number of iterations or if rate of change is
    # very small
    # YOUR CODE HERE
    initialize(img)
    max_iter = 12
    max_change_rate = 0.02
    dist = sys.float_info.max
    clustermask = np.zeros((height, width, 1), np.uint8)
    result = np.zeros((height, width, 3), np.uint8)

    i = 0
    lastDistance = 1
    iteration = 0
    change = 1.0
    while change > max_change_rate and i < max_iter:
        dist = assign_to_current_mean(img, result, clustermask)
        change = abs((lastDistance - dist) / lastDistance)
        update_mean(img, clustermask)

        print('Change rate: ', change)
        print('Error: ', dist)

        iteration += 1
        lastDistance = dist

    return result


# num of cluster
numclusters = 3
# corresponding colors for each cluster
cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128]]
# cluster_colors = []
# initialize current cluster centers (i.e. the pixels that represent a cluster center)
current_cluster_centers = np.zeros((numclusters, 1, 3), np.uint8)

# load image
imgraw = cv2.imread('./images/Lenna.png')
# imgraw = cv2.imread('./images/Lenna_HSV.png')
# imgraw = cv2.imread('./images/Lenna_LAB.png')
scaling_factor = 0.5
imgraw = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

print([(imgraw[15][15])[0], (imgraw[15][15])[1], (imgraw[15][15])[2]])

# compare different color spaces and their result for clustering
# YOUR CODE HERE or keep going with loaded RGB colorspace img = imgraw
image = imgraw

# execute k-means over the image
# it returns a result image where each pixel is color with one of the cluster_colors
# depending on its cluster assignment
# res = kmeans(image)
height, width = image.shape[:2]

res = kmeans(image)


h1, w1 = res.shape[:2]
h2, w2 = image.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
vis[:h1, :w1] = res
vis[:h2, w1:w1 + w2] = image

cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
