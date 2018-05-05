import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.178.21:4747/mjpegfeed")
sift = cv2.xfeatures2d.SIFT_create()
cv2.namedWindow('Interactive Systems: Towards AR Tracking')

while True:
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    # https://github.com/opencv/opencv/issues/6487
    img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Sift", img)

cv2.destroyAllWindows()






