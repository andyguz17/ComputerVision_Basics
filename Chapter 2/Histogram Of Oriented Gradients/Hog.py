import numpy as np
from skimage import exposure 
from skimage import feature
import cv2
 
# Lets initialize the HOG descriptor
hog = cv2.HOGDescriptor()

#We set the hog descriptor as a People detector
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img = cv2.imread('test_e.jpg')

#The image is pretty big so we will gibve it a resize
imX = 720
imY = 1080
img = cv2.resize(img,(imX,imY))

#We will define de 8x8 blocks in the winStride
boxes, weights = hog.detectMultiScale(img, winStride=(8,8))
boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])


for (xA, yA, xB, yB) in boxes:
    
    #Center in X 
    medX = xB - xA 
    xC = int(xA+(medX/2)) 

    #Center in Y
    medY = yB - yA 
    yC = int(yA+(medY/2)) 

    #Draw a circle in the center of the box 
    cv2.circle(img,(xC,yC), 1, (0,255,255), -1)

    # display the detected boxes in the original picture
    cv2.rectangle(img, (xA, yA), (xB, yB),
                        (255, 255, 0), 2)    

cv2.imshow('frame_2',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

