import cv2
from Aufgabe3.ImageStitcher import *

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.178.21:4747/mjpegfeed")
# cap = cv2.VideoCapture("http://141.64.175.37:4747/mjpegfeed")
sift = cv2.xfeatures2d.SIFT_create()

img_marker = cv2.imread('images/marker.jpg')
img_marker_gray = cv2.cvtColor(img_marker, cv2.COLOR_BGR2GRAY)
(h1, w1) = img_marker.shape[:2]

imageStitcher = ImageStitcher([img_marker])

kp_marker, dsc_marker = sift.detectAndCompute(img_marker_gray, None)

while True:
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (h1, w1))

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # keypoint matching
    kp_frame, dsc_frame = sift.detectAndCompute(gray, None)
    M = imageStitcher.match_keypoints(kp_frame, kp_marker, dsc_frame, dsc_marker)
    H, status, matches = M
    if status is not None:
        print(M)
        matchedImage = imageStitcher.draw_matches(img_marker, frame, kp_frame, kp_marker, matches, status)
        cv2.imshow("Sift", matchedImage)
    # M = None ????

cv2.destroyAllWindows()
