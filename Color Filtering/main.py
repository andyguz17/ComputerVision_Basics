import cv2

#Import the numpy library which will help with some matrix operations
import numpy as np 

image = cv2.imread('Filtering.png')

#I resized the image so it can be easier to work with
image = cv2.resize(image,(300,300))

#Once we read the image we need to change the color space to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

"""
Hsv limits are defined
here is where you define the range of the color you´re looking for
each value of the vector corresponds to the H,S & V values respectively
"""
min_green = np.array([50,220,220])
max_green = np.array([60,255,255])

min_red = np.array([170,220,220])
max_red = np.array([180,255,255])

min_blue = np.array([110,220,220])
max_blue = np.array([120,255,255])

"""
This is the actual color detection 
Here we will create a mask that contains only the colors defined in your limits
This mask has only one dimention, so its black and white 
"""
mask_g = cv2.inRange(hsv, min_green, max_green)
mask_r = cv2.inRange(hsv, min_red, max_red)
mask_b = cv2.inRange(hsv, min_blue, max_blue)

#We use the mask with the original image to get the colored post-processed image
res_b = cv2.bitwise_and(image, image, mask= mask_b)
res_g = cv2.bitwise_and(image,image, mask= mask_g)
res_r = cv2.bitwise_and(image,image, mask= mask_r)

cv2.imshow('Green',res_g)
cv2.imshow('Red',res_r)
cv2.imshow('Blue',res_b)

cv2.waitKey(0)
cv2.destroyAllWindows()