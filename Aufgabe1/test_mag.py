import numpy as np
import cv2
# https://stackoverflow.com/questions/19815732/what-is-gradient-orientation-and-gradient-magnitude

# normaize the grayscale image
def nm(img):
   normalizedImg = np.zeros(img.shape)
   return cv2.normalize(img, normalizedImg, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# load the image in grayscale
img = cv2.imread('Lenna.png', 0).astype(np.float32)

# sobel filter
sobelH = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float)
sobelV = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=np.float)

# horizontal gradient
sh = cv2.filter2D(img, -1, sobelH)
# vertical gradient
sv = cv2.filter2D(img, -1, sobelV)
# the magnitude of the gradient
sm = np.sqrt(sh**2 + sv**2).astype(np.uint8)

# central difference
centralH = np.array([[0,0,0],[-0.5,0,0.5],[0,0,0]],dtype=np.float)
centralV = np.array([[0,-0.5,0],[0,0,0],[0,0.5,0]],dtype=np.float)

# horizontal gradient
ch = cv2.filter2D(img, -1, centralH)
# vertical gradient
cv = cv2.filter2D(img, -1, centralV)
# the magnitude of the gradient
cm = np.sqrt(ch**2 + cv**2).astype(np.uint8)

# display the image
cv2.imshow('Images', np.vstack([np.hstack([nm(img), nm(sh), nm(sv), nm(sm)]), np.hstack([nm(img), nm(ch), nm(cv), nm(cm)])]))
cv2.waitKey(0)
cv2.destroyAllWindows()