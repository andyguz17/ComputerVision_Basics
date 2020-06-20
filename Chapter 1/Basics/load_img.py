#Import the Opencv Library
import cv2

#Read the image file
img = cv2.imread('test_image_1.jpg')

#Display the image in a window
cv2.imshow('image',img)

#Save the image "img" in the current path 
cv2.imwrite('image.jpg'.img)

#The window will close after a key press
cv2.waitKey(0)
cv2.destroyAllWindows()