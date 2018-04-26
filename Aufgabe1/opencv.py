import numpy as np
import cv2

img_grey = cv2.imread('images/Lenna.png', 0)
img_color = cv2.imread('images/Lenna.png', 1)

# https://stackoverflow.com/questions/40119743/convert-a-grayscale-image-to-a-3-channel-image
img_grey_converted = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)
# http://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/
lenna_grey_colored = np.hstack((img_grey_converted, img_color))
rows, cols = lenna_grey_colored.shape[:2]

cv2.imshow('image', lenna_grey_colored)

translation = 0
translation_matrix = np.float32([[1, 0, translation], [0, 1, 0]])
print(translation_matrix)
angle = 10
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
print(rotation_matrix)
M = np.hstack((rotation_matrix, translation_matrix))
print(M)

while True:
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    if cv2.waitKey(100) == ord('t'):
        translation += 10
        translation_matrix = np.float32([[1, 0, translation], [0, 1, 0]])
    if cv2.waitKey(100) == ord('z'):
        translation -= 10
        translation_matrix = np.float32([[1, 0, translation], [0, 1, 0]])
    if cv2.waitKey(100) == ord('r'):
        angle += 10
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    newImage = cv2.warpAffine(lenna_grey_colored, translation_matrix, (cols, rows))
    newImage = cv2.warpAffine(newImage, rotation_matrix, (cols, rows))
    cv2.imshow('image', newImage)

cv2.destroyAllWindows()
