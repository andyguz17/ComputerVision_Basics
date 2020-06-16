import cv2
import numpy as np

img = cv2.imread('test_img_b.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(450,350))

#Here we define the sobel operators
#This are no more than a numpy matrix
kernel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
kernel_y = np.array([[-1,0,1],[-2,-0,2],[-1,0,1]])

#This part is where the magic happens
#We convolve the image read with the kernels defined
x_conv = cv2.filter2D(img,-1,kernel_x)
y_conv = cv2.filter2D(img,-1,kernel_y)

cv2.imshow('Original',img)
cv2.imshow('sobelx',x_conv)
cv2.imshow('sobely',y_conv)

cv2.imwrite('sobelx_b.jpg',x_conv)
cv2.imwrite('sobely_b.jpg',y_conv)

cv2.waitKey(0)
cv2.destroyAllWindows()