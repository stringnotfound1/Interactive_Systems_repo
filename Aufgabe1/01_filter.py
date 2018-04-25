import numpy as np
import cv2


def im2double(im):
    """
    Converts uint image (0-255) to double image (0.0-1.0) and generalizes
    this concept to any range.

    :param im:
    :return: normalized image
    """
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    k = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)
    return k / np.sum(k)


def convolution_2d(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix - 3x3, or 5x5 matrix
    :return: result of the convolution
    """
    # TODO write convolution of arbritrary sized convolution here
    # Hint: you need the kernelsize

    height, width = img.shape[:2]
    offset = int(kernel.shape[0] / 2)
    newimg = np.zeros(img.shape)

    img = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_REPLICATE)

    # slide kernel across image
    for y in np.arange(offset, height + offset):
        for x in np.arange(offset, width + offset):
            # extract Region of Interest
            roi = img[y - offset:y + offset + 1, x - offset:x + offset + 1]
            convoluted = (roi * kernel).sum()

            # store the convolved value in the new img (x,y)
            newimg[y - offset, x - offset] = convoluted

    return newimg


if __name__ == "__main__":
    # 1. load image in grayscale
    img_grey = cv2.imread('Lenna.png', 0)
    # 2. convert image to 0-1 image (see im2double)
    img_grey = im2double(img_grey)

    # image kernels
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gk = make_gaussian(11)

    # 3 .use image kernels on normalized image
    img_grey = convolution_2d(img_grey, gk)
    sobel_x = convolution_2d(img_grey, sobelmask_x)
    sobel_y = convolution_2d(img_grey, sobelmask_y)
    # 4. compute magnitude of gradients
    mag_no_threshold = np.sqrt((sobel_x ** 2 + sobel_y ** 2).astype(np.float))
    mag_no_threshold = cv2.normalize(mag_no_threshold, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    mag = np.sqrt((sobel_x ** 2 + sobel_y ** 2).astype(np.float))
    # normalize array
    mag *= 255.0 / mag.max()
    # threshold
    mag[mag < 50] = 0
    # Show resulting images
    cv2.imshow("sobel_x", sobel_x)
    cv2.imshow("sobel_y", sobel_y)
    cv2.imshow("mog", mag_no_threshold)
    cv2.imshow("mog_threshold", mag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
