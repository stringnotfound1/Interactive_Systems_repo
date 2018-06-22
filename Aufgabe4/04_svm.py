import numpy as np
import cv2
import glob
from sklearn import svm


############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################

def create_keypoints(w, h):
    keypointSize = 15
    keypoints_local = []
    # TODO change factor to 1, still works with 30+
    factor = 30
    # factor = 1

    for x in np.arange(keypointSize, h - keypointSize, factor):
        for y in np.arange(keypointSize, w - keypointSize, factor):
            keypoint_local = cv2.KeyPoint(x, y, keypointSize)
            keypoints_local.append(keypoint_local)

    return keypoints_local


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px
# images = glob.glob('./images/db/train/**/*.jpg')
# image_list = []
# for file_name in images:
#     image_list.append(cv2.imread(file_name, 1))
#
# print(len(image_list))

sift = cv2.xfeatures2d.SIFT_create()
keypoints = create_keypoints(256, 256)

# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers

labels = {1: 'car', 2: 'face', 3: 'flower'}
train = []
print("Get images and assign labels and descriptors")
for label_int, label_caption in labels.items():
    image_paths = glob.glob('./images/db/train/{}/*.jpg'.format(label_caption + 's'))
    for image_path in image_paths:
        img = cv2.imread(image_path, 1)
        kp, des = sift.compute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), keypoints)
        # ravel creates 1-D array
        train.append((label_int, des.ravel(), img))

print("Images with labels and descriptors finished")

y_train = train[0][0]
X_train = np.asmatrix(train[0][1])
for y, X, _ in train[1:]:
    X_train = np.vstack((X_train, X))
    y_train = np.vstack((y_train, y))

print(X_train.shape)
print(y_train)

# 3. We use scikit-learn to train a SVM classifier. Specifically we use a LinearSVC in our case. Have a look at the
# documentation. You will need .fit(X_train, y_train)
print('Training SVM started')
classifier = svm.LinearSVC()
classifier.fit(X_train, y_train.ravel())
print('Training SVM finished')

# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image

test_images = glob.glob('./images/db/test/*.jpg')
for test_img_path in test_images:
    test_img = cv2.imread(test_img_path, 1)
    _, test_dsc = sift.compute(test_img, keypoints)
    test_dsc = test_dsc.ravel()
    prediction = classifier.predict([test_dsc])
    print("Image {} seems to be part of category {}, which means it is a {}.".format(test_img_path, prediction[0], labels.get(prediction[0])))

# 5. output the class + corresponding name
