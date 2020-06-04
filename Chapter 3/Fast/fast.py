import cv2
import numpy as np

image = cv2.imread('corner_test_2.png')

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

fast = cv2.FastFeatureDetector_create() 

#Keypoints using non Max Supression
Keypoints_1 = fast.detect(gray, None)

#Set non Max Supression disabled 
fast.setNonmaxSuppression(False)

#Keypoints without non max Suppression
Keypoints_2 = fast.detect(gray, None)

#Create tywo instances of the original image
image_with_nonmax = np.copy(image)
image_without_nonmax = np.copy(image)

# Draw keypoints on top of the input image
cv2.drawKeypoints(image, Keypoints_1, image_with_nonmax, color=(0,35,250))
cv2.drawKeypoints(image, Keypoints_2, image_without_nonmax, color=(0,35,250))

cv2.imshow('Non max supression',image_with_nonmax)
cv2.imshow('without non max Supression',image_without_nonmax)

cv2.waitKey(0)
cv2.destroyAllWindows()
