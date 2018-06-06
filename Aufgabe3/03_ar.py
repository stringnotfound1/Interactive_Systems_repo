import cv2
import numpy as np

# global constants
min_matches = 10
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)

# initialize flann and SIFT extractor
# note unfortunately in the latest OpenCV + python is a minor bug in the flann
# flann = cv2.FlannBasedMatcher(indexParams, {})
# so we use the alternative but slower Brute-Force Matcher BFMatcher
# YOUR CODE

sift = cv2.xfeatures2d.SIFT_create()
bfMatcher = cv2.BFMatcher()

# extract marker descriptors
# YOUR CODE
img_marker = cv2.imread('images/marker.jpg', 1)
kp_marker, dsc_marker = sift.detectAndCompute(img_marker, None)


def render_virtual_object(img, x_start, y_start, x_end, y_end, quad):
    # define vertices, edges and colors of your 3D object, e.g. cube

    # YOUR CODE HERE
    z = 0.3
    vertices = np.float32([[0, 0, 0],
                           [1, 0, 0],
                           [1, 1, 0],
                           [0, 1, 0],
                           [0, 0, z],
                           [1, 0, z],
                           [1, 1, z],
                           [0, 1, z]])

    edges = [(0, 1),
             (1, 2),
             (2, 3),
             (3, 0),

             (0, 4),
             (1, 5),
             (2, 6),
             (3, 7),

             (4, 5),
             (5, 6),
             (6, 7),
             (7, 4)]

    color_lines = (150, 0, 0)

    # define quad plane in 3D coordinates with z = 0
    quad_3d = np.float32([[x_start, y_start, 0], [x_end, y_start, 0],
                          [x_end, y_end, 0], [x_start, y_end, 0]])

    h, w = img.shape[:2]
    # define intrinsic camera parameter
    K = np.float64([[w, 0, 0.5 * (w - 1)],
                    [0, w, 0.5 * (h - 1)],
                    [0, 0, 1.0]])

    # find object pose from 3D-2D point correspondences of the 3d quad using Levenberg-Marquardt optimization
    # in order to work we need K (given above and YOUR distortion coefficients from Assignment 2 (camera calibration))
    # YOUR VALUES HERE
    # [[ 0.06489938  0.2053827   0.00292677  0.00300208 -1.12425709]]
    dist_coef = np.array([0.06489938,  0.2053827,   0.00292677,  0.00300208, -1.12425709])
    # dist_coef = None

    # compute extrinsic camera parameters using cv2.solvePnP
    # YOUR CODE HERE
    _, rvec, tvec = cv2.solvePnP(quad_3d, quad, K, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)
    # transform vertices: scale and translate form 0 - 1, in window size of the marker
    scale = [x_end - x_start, y_end - y_start, x_end - x_start]
    trans = [x_start, y_start, -x_end - x_start]

    verts = scale * vertices + trans

    # call cv2.projectPoints with verts, and solvePnP result, K, and dist_coeff
    # returns a tuple that includes the transformed vertices as a first argument
    # YOUR CODE HERE
    verts, _ = cv2.projectPoints(verts, rvec, tvec, K, dist_coef)
    # we need to reshape the result of projectPoints
    verts = verts.reshape(-1, 2)

    # render edges
    for i, j in edges:
        (x_start, y_start), (x_end, y_end) = verts[i], verts[j]
        cv2.line(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), color_lines, 2)


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.178.21:4747/mjpegfeed")
cv2.namedWindow('Interactive Systems: AR Tracking')
while True:
    # YOUR CODE

    # detect and compute descriptor in camera image
    # and match with marker descriptor
    _, frame = cap.read()
    kp_frame, dsc_frame = sift.detectAndCompute(frame, None)
    if dsc_marker is None or dsc_frame is None:
        continue
    matches = bfMatcher.knnMatch(dsc_frame, dsc_marker, 2)
    # filter matches by distance [Lowe2004]
    matches = [match[0] for match in matches if len(match) == 2 and
               match[0].distance < match[1].distance * 0.75]

    # if there are less than min_matches we just keep going looking
    # early break
    if len(matches) < min_matches:
        cv2.imshow('Interactive Systems: AR Tracking', frame)
        continue

    # extract 2d points from matches data structure
    p0 = [kp_marker[m.trainIdx].pt for m in matches]
    p1 = [kp_frame[m.queryIdx].pt for m in matches]
    # transpose vectors
    p0, p1 = np.array([p0, p1])

    # we need at least 4 match points to find a homography matrix
    if len(p0) < 4:
        cv2.imshow('Interactive Systems: AR Tracking', frame)
        continue

    # find homography using p0 and p1, returning H and status
    # H - homography matrix
    # status - status about inliers and outliers for the plane mapping
    # YOUR CODE
    # (H, status) = cv2.findHomography(p0, p1, cv2.RANSAC, 4.0)
    (H, mask) = cv2.findHomography(p0, p1, cv2.RANSAC, 4.0)
    # on the basis of the status object we can now filter RANSAC outliers
    if mask is None:
        continue
    mask = mask.ravel() != 0
    if mask.sum() < min_matches:
        cv2.imshow('Interactive Systems: AR Tracking', frame)
        continue

    # take only inliers - mask of Outlier/Inlier
    p0, p1 = p0[mask], p1[mask]
    # get the size of the marker and form a quad in pixel coords np float array using w/h as the corner points
    # YOUR CODE HERE
    h1, w1 = markerImg.shape[:2]
    quad = [np.array([0, 0], dtype=np.float32),
            np.array([0, h1], dtype=np.float32),
            np.array([w1, h1], dtype=np.float32),
            np.array([w1, 0], dtype=np.float32),    # bottom left
            ]
    print(quad)
    # perspectiveTransform needs a 3-dimensional array
    quad = np.array([quad])
    quad_transformed = cv2.perspectiveTransform(quad, H)
    # transform back to 2D array
    quad = quad_transformed[0]

    # render quad in image plane and feature points as circle using cv2.polylines + cv2.circle
    # (G,B, R) ???
    cv2.polylines(frame, [quad.astype(dtype=np.int)], isClosed=True, color=(0, 0, 255), thickness=2)
    for fp1 in p1:
        fp1 = fp1.astype(np.int)
        cv2.circle(frame, (fp1[0], fp1[1]), 10, (0, 255, 0))

    # render virtual object on top of quad
    render_virtual_object(frame, 0, 0, h1, w1, quad)

    cv2.imshow('Interactive Systems: AR Tracking', frame)
    key = cv2.waitKey(15) & 0xFF
    if key == ord('q'):
        break
