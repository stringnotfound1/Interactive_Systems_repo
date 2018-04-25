import numpy as np
import cv2

img_grey = cv2.imread('Lenna.png', 0)
rows, cols = img_grey.shape[:2]

# https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
kernel_blur = np.array(([0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]))
kernel_sobel_x = np.array(([1, 0, -1], [2, 0, -2], [1, 0, -1]))
kernel_sobel_y = np.array(([1, 2, 1], [0, 0, 0], [-1, -2, -1]))

blur = cv2.filter2D(img_grey, -1, kernel_blur)
# print(blur)

sobel_x_float = cv2.filter2D(blur, -1, kernel_sobel_x).astype(np.float)

sobel_y_float = cv2.filter2D(blur, -1, kernel_sobel_y).astype(np.float)

dst = img_grey

mag_no_threshold = np.sqrt((sobel_x_float**2 + sobel_y_float**2).astype(np.float))
mag_no_threshold = cv2.normalize(mag_no_threshold, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
mag = np.sqrt((sobel_x_float**2 + sobel_y_float**2).astype(np.float))
mag = cv2.filter2D(mag, -1, kernel_blur)
# normalize array
mag *= 255.0/mag.max()
# threshold
mag[mag < 50] = 0

sobel_x_float = cv2.normalize(sobel_x_float, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
sobel_y_float = cv2.normalize(sobel_y_float, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

while True:
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    if cv2.waitKey(100) & 0xFF == ord('b'):
        dst = blur
    if cv2.waitKey(100) & 0xFF == ord('x'):
        dst = sobel_x_float
    if cv2.waitKey(100) & 0xFF == ord('y'):
        dst = sobel_y_float
    if cv2.waitKey(100) & 0xFF == ord('v'):
        dst = img_grey
    if cv2.waitKey(100) & 0xFF == ord('m'):
        dst = mag
    if cv2.waitKey(100) & 0xFF == ord('n'):
        dst = mag_no_threshold

    cv2.imshow('image', dst)

