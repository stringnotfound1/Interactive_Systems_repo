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
    # compute x and y gradients
    sobel_x = cv2.Sobel(img_grey, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img_grey, cv2.CV_64F, 0, 1, ksize=5)
    # compute magnitude and angle of the gradients
    mag = np.sqrt((sobel_x ** 2 + sobel_y ** 2).astype(np.float))
    phase = cv2.phase(sobel_x, sobel_y, True)
    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        print(kp.pt, kp.size)
        # extract angle in keypoint sub window
        # extract gradient magnitude in keypoint subwindow
        kp_pt_x = (int(kp.pt[0]))
        kp_pt_y = (int(kp.pt[1]))
        kp_size = (int(kp.size))
        image_part = img_grey[kp_pt_x-kp_size:kp_pt_x+kp_size, kp_pt_y-kp_size:kp_pt_y+kp_size]
        sobel_x_part = cv2.Sobel(image_part, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y_part = cv2.Sobel(image_part, cv2.CV_64F, 0, 1, ksize=5)
        # mag = np.sqrt((sobel_x_part ** 2 + sobel_y_part ** 2).astype(np.float))
        mag = cv2.magnitude(sobel_x_part, sobel_y_part)
        # print("Magnitude:", mag)
        phase = cv2.phase(sobel_x_part, sobel_y_part, True)
        # print(phase)
        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        # if (mag != 0):
        print(phase)
        (hist, bins) = np.histogram(phase, bins=[0, 20, 40, 60, 80, 100])

        plot_histogram(hist, bins)
        #
        descr[count] = hist

    return descr


keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
test = cv2.imread('./images/hog_test/diag.jpg')
descriptor = compute_simple_hog(test, keypoints)

