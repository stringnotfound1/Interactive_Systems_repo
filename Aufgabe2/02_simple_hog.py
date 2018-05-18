import numpy as np
import cv2
import math
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


###############################################################
#
# Write your own descriptor / Histogram of Oriented Gradients
#
###############################################################


def plot_histogram(hist, bins):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def compute_simple_hog(imgcolor, keypoints):
    # convert color to gray image and extract feature in gray
    img_grey = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)
    img_grey = np.float32(img_grey) / 255.0
    # compute x and y gradients
    sobel_x = cv2.Sobel(img_grey, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_grey, cv2.CV_64F, 0, 1, ksize=3)
    phase = cv2.phase(sobel_x, sobel_y, True)
    mag = cv2.magnitude(sobel_x, sobel_y)
    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 7), np.float32)
    # descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        print(kp.pt, kp.size)
        # extract angle in keypoint sub window
        # extract gradient magnitude in keypoint subwindow
        kp_pt_x = (int(kp.pt[0]))
        kp_pt_y = (int(kp.pt[1]))
        kp_size = (int(kp.size / 2))
        sub_phase = phase[kp_pt_y - kp_size:kp_pt_y + kp_size, kp_pt_x - kp_size:kp_pt_x + kp_size]
        sub_mag = mag[kp_pt_x - kp_size:kp_pt_x + kp_size, kp_pt_y - kp_size:kp_pt_y + kp_size]
        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        #         # answer to that question use np.histogram
        print(sub_phase)
        (hist, bins) = np.histogram(sub_phase[sub_mag > 0], bins=[0, 1, 2, 3, 4, 5, 6, 7])

        plot_histogram(hist, bins)
        #
        descr[count] = hist

    return descr


keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
# test = cv2.imread('./images/hog_test/horiz.jpg')
test = cv2.imread('./images/hog_test/diag.jpg')
# test = cv2.imread('./images/hog_test/circle.jpg')
descriptor = compute_simple_hog(test, keypoints)
