import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# Code from:
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

# Prepared for M1IV students, by Prof. Slimane Larabi
#===================================================
# The code is modified for the specific database
#===================================================
# Function to get the names of the images from the "test" folder in a list:
def create_image_path_list(directory_path):
    image_path_list = []
    if not os.path.isdir(directory_path):
        print(f"The specified path '{directory_path}' is not a valid directory.")
        return image_path_list

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path_list.append(os.path.join(directory_path, filename).replace('\\', '/'))

    return image_path_list
#===================================================

# From the Test folder we cropped an image [plaque.png] to search in the dataset [test/valid]
img1 = cv2.imread('plaque.png', 0)  # queryImage

# To get the images from the database
ListImages = create_image_path_list('test/valid')

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)

#
threshold = 0.75 #it has been found to effectively filter out false matches while retaining good ones
#note:. Lowering the threshold might result in fewer but more accurate matches, while increasing it might include more matches but also more false positives.
previous = 0
# To find the best matching image
for image in ListImages:
    img2 = cv2.imread(image, 0)  # trainImage
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])

    if len(good) > previous:
        previous_list = good
        previous = len(good)
        img_ideal = img2
        kp_ideal = kp2
        des_ideal = des2

img3 = cv2.drawMatchesKnn(img1, kp1, img_ideal, kp_ideal, good, None, flags=2)

plt.imshow(img3), plt.show()
