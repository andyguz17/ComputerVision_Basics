import cv2
import numpy as np

image = cv2.imread('test_e.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

fast = cv2.FastFeatureDetector_create() 
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

keypoints = fast.detect(gray, None)    
brief_keypoints, descriptor = brief.compute(gray, keypoints)

brief = np.copy(image)
non_brief = np.copy(image)

# Draw keypoints on top of the input image
cv2.drawKeypoints(image, brief_keypoints, brief, color=(250,35,20))
cv2.drawKeypoints(image, keypoints, non_brief, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Keypoints after BRIEF',brief)
cv2.imshow('Keypoints Before BRIEF',non_brief)

cv2.waitKey(0)
cv2.destroyAllWindows()
 