import random
from collections import defaultdict

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
    squared_distance = 0

    # Assuming correct input to the function where the lengths of two features are the same

    for i in range(len(a)):
        squared_distance += (a[i] - b[i]) ** 2

    e_distance = math.sqrt(squared_distance)

    return e_distance

# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the error

def update_mean(img, clustermask):
    """This function should compute the new cluster center, i.e. numcluster mean colors"""
    cluster_color_list = defaultdict(list)
    print("update_means")
    for h in range(0, h1):
        for w in range(0, w1):
            cluster_id = int(clustermask[h][w])
            color_at_pixel = img[h][w]
            cluster_color_list[cluster_id].append(color_at_pixel)
        for k in cluster_color_list.keys():
            current_cluster_centers[k] = np.uint8(np.mean(cluster_color_list[k], axis=0))

def assign_to_current_mean(img, result, clustermask):
    """The function expects the img, the resulting image and a clustermask.
    After each call the pixels in result should contain a cluster_color corresponding to the cluster
    it is assigned to. clustermask contains the cluster id (int [0...num_clusters]
    Return: the overall error (distance) for all pixels to there closest cluster center (mindistance px - cluster center).
    """
    overall_dist = 0
    for h in range(0, h1):
        for w in range(0, w1):
            color_pixel = img[h][w]
            color_mean = current_cluster_centers[clustermask[h][w]]
            color_mean = np.reshape(color_mean, color_pixel.shape)
            overall_dist += distance(color_pixel, color_mean)
            result[h][w] = color_mean
    return overall_dist



def initialize(img):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""
    # YOUR CODE HERE
    for i in range(0, numclusters):
        rng_h = int(h1 * np.random.random())
        rng_w = int(w1 * np.random.random())
        current_cluster_centers[i] = img[rng_h][rng_w]

def kmeans(img):
    """Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    max_iter = 10
    max_change_rate = 0.02
    dist = sys.float_info.max

    clustermask = np.zeros((h1, w1, 1), np.uint8)
    result = np.zeros((h1, w1, 3), np.uint8)

    # initializes each pixel to a cluster
    # iterate for a given number of iterations or if rate of change is
    # very small
    # YOUR CODE HERE
    i = 0
    change_rate = 1.0
    while i < max_iter and change_rate > max_change_rate:
        changes = 0
        for h in range(0, h1):
            for w in range(0, w1):
                # get pixel and calculate distances to all current cluster centers
                pixel = img[h][w]
                currentClusterID = clustermask[h][w]
                distances = []
                # I don´t use PriorityQueue here since it crashes when to values have the same priority
                # (see last assignment, gave an additional index to the queue to have a second index
                # for it to order by).
                for id, cc in enumerate(current_cluster_centers):
                    cc = np.reshape(cc, pixel.shape)
                    distances.append((distance(cc, pixel), id))

                closestClusterID = sorted(distances, key=lambda x: x[0])[0][1]  # get closest clusterID
                if closestClusterID != currentClusterID:
                    # Set id of closest cluster center to clustermask at this position
                    clustermask[h][w] = closestClusterID
                    changes += 1

        change_rate = changes / (w1 * h1)
        print('Change rate: ', change_rate)
        overallError = assign_to_current_mean(img, result, clustermask)
        update_mean(img, clustermask)
        print('Error: ', overallError)
        i += 1
    return result


# num of cluster
numclusters = 3
# corresponding colors for each cluster
cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128]]
# initialize current cluster centers (i.e. the pixels that represent a cluster center)
current_cluster_centers = np.zeros((numclusters, 1, 3), np.float32)

# load image
imgraw = cv2.imread('./images/Lenna.png')
# imgraw = cv2.imread('./images/Lenna_HSV.png')
# imgraw = cv2.imread('./images/Lenna_LAB.png')
scaling_factor = 0.5
imgraw = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

# compare different color spaces and their result for clustering
# YOUR CODE HERE or keep going with loaded RGB colorspace img = imgraw
image = imgraw
h1, w1 = image.shape[:2]
initialize(image)

# execute k-means over the image
# it returns a result image where each pixel is color with one of the cluster_colors
# depending on its cluster assignment
# res = kmeans(image)
res = kmeans(image)

h1, w1 = res.shape[:2]
h2, w2 = image.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
vis[:h1, :w1] = res
vis[:h2, w1:w1 + w2] = image

cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
