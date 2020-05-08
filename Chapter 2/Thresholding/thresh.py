import cv2
import numpy as np

img = cv2.imread('test_image.png',0)
img = cv2.resize(img,(450,450))

ret,thresh1 = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

cv2.imshow('Original',img)
cv2.imshow('thresh1',thresh1)
cv2.imshow('thresh2',thresh2)
cv2.imshow('thresh3',thresh3)
cv2.imshow('thresh4',thresh4)
cv2.imshow('thresh5',thresh5)

cv2.waitKey(0)
cv2.destroyAllWindows()