import numpy as np
import cv2

#First we need to upload the xml classifier 
face_cascade = cv2.CascadeClassifier('frontalface.xml')

img = cv2.imread('face.jpg')
img_2 = cv2.imread('many.jpg')

img = cv2.resize(img,(400,700))

#We need to convert  the image to grayScale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

#The Scale Factor determines just how much the original image
#is going to be reduced in every Scale
ScaleFactor = 1.2

#Just like the name says this wil determine the number of neighbors 
#to a higher value, the model will be more selective 
minNeighbors = 4

#We apply the cascades too our grayscaled images
faces = face_cascade.detectMultiScale(gray, ScaleFactor, minNeighbors)
faces_2 = face_cascade.detectMultiScale(gray_2, ScaleFactor, minNeighbors)

#Once the algorithm finished we will extract the coordinates of the detections 
for (x,y,w,h) in faces:
    #For every coordinate we will draw a rectangle in the original image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    roi_color = img[y:y+h, x:x+w]

#same for the second image 
for (x,y,w,h) in faces_2:
    cv2.rectangle(img_2,(x,y),(x+w,y+h),(255,255,0),2)
    roi_color = img_2[y:y+h, x:x+w]  

cv2.imshow('Face',img)
cv2.imshow('Faces',img_2)

cv2.imwrite('face_1.jpg',img)
cv2.imwrite('faces_2.jpg',img_2)

cv2.waitKey(0)
cv2.destroyAllWindows()
