import numpy as np
import cv2
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()
img_2 = cv2.imread('test_b.jpg')

#Size for the image 
imX = 1280
imY = 720

img_2 = cv2.resize(img_2,(imX,imY))

gray_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)

boxes_2, weights_2 = hog.detectMultiScale(img_2, winStride=(8,8) )
boxes_2 = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes_2])


for (xA, yA, xB, yB) in boxes_2:
    
    #Center in X 
    medX = xB - xA 
    xC = int(xA+(medX/2)) 

    #Center in Y
    medY = yB - yA 
    yC = int(yA+(medY/2)) 

    #Draw a circle in the center of the box 
    cv2.circle(img_2,(xC,yC), 1, (0,255,255), -1)

    # display the detected boxes in the colour picture
    cv2.rectangle(img_2, (xA, yA), (xB, yB),
                        (255, 255, 0), 2)    


cv2.imshow('frame_2',img_2)

cv2.waitKey(0)
cv2.destroyAllWindows()

