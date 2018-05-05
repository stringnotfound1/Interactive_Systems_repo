import numpy as np
import cv2

cap = cv2.VideoCapture("http://192.168.178.21:4747/mjpegfeed")
# cap = cv2.VideoCapture(0)
sift = cv2.xfeatures2d.SIFT_create()
blur_flag = False

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    if cv2.waitKey(100) == ord('b'):
        print("BLUR")
        blur_flag = not blur_flag

    if blur_flag:
        kernel = np.ones((7, 7), np.float32) / 49
        gray = cv2.filter2D(gray, -1, kernel)

    # Display the resulting frame
    cv2.imshow('frame', gray)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()