import cv2
import numpy as np

img = cv2.imread('border.png')

#Convert the image to gray scale so the gradient is better visible
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Apply the horizontal sobel operator with a kernel size of 3
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)

#Apply the vertical sobel operator with a kernel size of 3
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

cv2.imshow('Original',img)
cv2.imshow('sobelx',sobelx)
cv2.imshow('sobely',sobely)

cv2.imwrite('x.jpg', sobelx)
cv2.imwrite('y.jpg', sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()