import cv2
import numpy as np

def detect_object_in_scene(object_image_path, scene_image_path):
    """
    Detects an object in a scene using SIFT and homography.
    Args:
        object_image_path (str): Path to the object image.
        scene_image_path (str): Path to the scene image.
    """
    # Load the object and scene images in grayscale
    object_img = cv2.imread(object_image_path, cv2.IMREAD_GRAYSCALE)
    scene_img = cv2.imread(scene_image_path, cv2.IMREAD_GRAYSCALE)
    
    if object_img is None or scene_img is None:
        print("Error: Could not load one or both images.")
        return
    
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect and compute keypoints and descriptors
    keypoints_obj, descriptors_obj = sift.detectAndCompute(object_img, None)
    keypoints_scene, descriptors_scene = sift.detectAndCompute(scene_img, None)
    
    # Match descriptors using Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors_obj, descriptors_scene, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Draw matches and perform homography if enough good matches are found
    if len(good_matches) > 10:  # Minimum number of matches required for homography
        # Extract locations of matched keypoints
        src_pts = np.float32([keypoints_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Draw bounding box around the detected object
        h, w = object_img.shape
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        
        scene_img_with_box = cv2.polylines(scene_img.copy(), [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
        
        # Draw the matches
        matches_img = cv2.drawMatches(object_img, keypoints_obj, scene_img_with_box, keypoints_scene, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Show the result
        cv2.imshow("Good Matches & Object Detection", matches_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Not enough good matches found.")

# Example usage
detect_object_in_scene("plaque.png", "image.jpg")
