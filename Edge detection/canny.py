import cv2
import numpy as np 

img = cv2.imread('test_img.png')
img = cv2.resize(img,(450,350))

#The canny detector uses two parameters appart from the image:
#The minimum and maximum intensity gradient (30,100)
edges = cv2.Canny(img,30,100)
cv2.imshow('Original',img)
cv2.imshow('Edges',edges)

cv2.waitKey(0)
cv2.destroyAllWindows()