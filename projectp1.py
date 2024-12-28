# Import necessary libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Code sourced from the OpenCV tutorial on feature matching:
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

# Prepared for M1IV students by Prof. Slimane Larabi
#===================================================

# Read the query image and the model image (both in grayscale)
img1 = cv2.imread('plaque.png', 0)          # Query image
img2 = cv2.imread('image.jpg', 0)          # Train image

# Initiate the SIFT (Scale-Invariant Feature Transform) detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images using SIFT
kp1, des1 = sift.detectAndCompute(img1, None)  # Keypoints and descriptors for query image
kp2, des2 = sift.detectAndCompute(img2, None)  # Keypoints and descriptors for model image

# Print the descriptor of the first keypoint in the query image and its coordinates
print(des1[0], kp1[0].pt)

# Initialize the brute-force (BF) matcher with default parameters
bf = cv2.BFMatcher()

# Use the BFMatcher to find the best matches between descriptors in the two images
matches = bf.knnMatch(des1, des2, k=2)  # k=2 returns the two best matches for each descriptor
#This method finds the `k` nearest neighbors (here `k=2`) for each descriptor in `des1` from `des2`.

# Apply the ratio test (Lowe's ratio test) to filter out good matches
good = []  # List to store the good matches
for m, n in matches:  # m and n are the two closest matches for each keypoint
    if m.distance < 0.75 * n.distance:  # Lowe's ratio test: keep matches where the distance ratio is less than 0.75
        print(m.queryIdx, m.trainIdx, m.distance, n.queryIdx, n.trainIdx, n.distance, n.imgIdx)  # Print match info
        good.append([m])  # Add the good match to the list

# Draw the good matches between the two images using cv2.drawMatchesKnn
# Note that we need to pass a list of lists for the matches in this function
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# Display the result using matplotlib
plt.imshow(img3)  # Display the image with the drawn matches
plt.show()  # Show the plot
"""
- Applies a **ratio test** (Lowe's ratio test) to filter out ambiguous matches:
       - For each match, it compares the distance of the closest descriptor (`m.distance`) to the distance of the second closest descriptor (`n.distance`).
       - A match is retained if the closest match is significantly better (e.g., `m.distance < 0.75 * n.distance`).
     - This reduces false positives and ensures higher-quality matches.
"""