### Face detection 

In the computer vision world one of the most important task is the classification, where you will want to detect certain objects in a image. For long time we always wanted a way to detect people, nowadays almost every single camera has a feature that detects people faces and it's pretty accurate. It´s even used in some social apps like snapchat, where an algorithm detects a face and extracts it´s characteristics. 

A pretty good classifier algorithm that opencv has is the *Haar Cascade* which works with Haar Wavelets to analyze image pixels into squares, this was proposed by <a href="https://ieeexplore.ieee.org/abstract/document/990517">Viola & Jones</a> in 2001.This classifier work just like convolutions kernels, where we try to extract different features of the image with *"integral image"*. This uses the AdaBoost learning algorithm selecting small numbers of features from a large set of images.


The main idea is that you can train this models to detect any object you want, similar to convolutional neural networks. In the code ahead you will learn how to use a trained model to detect faces in images (of course you can download more models to play with). 

###### *FaceDetection/haar.py*

```python 
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
```
The Scale Factor determines just how much the original image is going to be reduced in every Scale

```python

ScaleFactor = 1.2
```
Just like the name says this wil determine the number of neighbors to a higher value, the model will be more selective 
```python 
minNeighbors = 3
```
We apply the cascades too our grayscaled images
```python 
faces = face_cascade.detectMultiScale(gray, ScaleFactor, minNeighbors)
faces_2 = face_cascade.detectMultiScale(gray_2, ScaleFactor, minNeighbors)

#Once the algorithm finished we will extract the coordinates of the detections 
for (x,y,w,h) in faces:
    #For every coordinate we will draw a rectangle in the original image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

    #This will be a cut of the area of interest
    roi = img[y:y+h, x:x+w]

#same for the second image 
for (x,y,w,h) in faces_2:
    cv2.rectangle(img_2,(x,y),(x+w,y+h),(255,255,0),2)
    roi = img_2[y:y+h, x:x+w]  

cv2.imshow('Face',img)

```

<div style="text-align:center"><img src="Resources/face.jpg" width = 20% /></div>
<br>

```Python
cv2.imshow('Faces',img_2)

```

<div style="text-align:center"><img src="Resources/faces.jpg" width = 40% /></div>
<br>

```Python

cv2.waitKey(0)
cv2.destroyAllWindows()

```
<a href="https://github.com/opencv/opencv/tree/master/data/haarcascades">Opencv</a> offers some pre-trained models, that are pretty fast and accurate, you can download the xml files and try them with some images you want. In personal opinion the face detection with Haar Cascades is really good, of course convolutional neural networks are more accurate but they use a lot of resources to work properly. A good approach if you want to use a face as a biometric patron,  would be to use a lightweight algorithm like Haar Cascades just to get the area of interest and then move this area to a convolutional neural network to get the face classifier. With this instead of analyzing all the image with CNN's you will analize  just a little area where you already known that a face is there.    