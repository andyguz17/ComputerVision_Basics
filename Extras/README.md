# arTags (Augmented Reality)

###### arTags/Generate.py

```Python
import os 
import cv2 
import numpy as np 
from cv2 import aruco

#Verify if the folder exists or not 
verify = os.path.isdir('./Tags')

if not(verify):
    #Create a folder to store the generated tags
    os.mkdir("./Tags")

#Initialize the dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

for i in range(1, 4):
    
    size = 700
    img = aruco.drawMarker(aruco_dict, i, size)
    
    cv2.imwrite('./Tags/image_'+str(i)+".jpg",img)
    
    cv2.imshow('asd',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
```

###### arTags/PPolygon.py 

```Python
import cv2 
import numpy as np 
from cv2 import aruco

def order_coordinates(pts):
    
    #Initialize an empty array to save to next values 
	coordinates = np.zeros((4, 2), dtype="int")

	s = pts.sum(axis=1)
	coordinates[0] = pts[np.argmin(s)]
	coordinates[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis=1)
	coordinates[1] = pts[np.argmin(diff)]
	coordinates[3] = pts[np.argmax(diff)]

	return coordinates

image = cv2.imread('./Examples/a3.jpg')
h, w = image.shape[:2]

image = cv2.resize(image,(int(w*0.7), int(h*0.7)))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Initialize the aruco Dictionary and its parameters 
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

#Detect the corners and id's in the examples 
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

#Show the markers detected
cv2.imshow('markers',frame_markers)

#Initialize an empty list for the coordinates 
params = []

for i in range(len(ids)):

    #Catch the corners of each tag
    c = corners[i][0]

    #Draw a circle in the center of each detection
    cv2.circle(image,(int(c[:, 0].mean()), int(c[:, 1].mean())), 3, (255,255,0), -1)
    
    #Save thhe center coordinates for each tag
    params.append((int(c[:, 0].mean()), int(c[:, 1].mean())))

#Convert the coordinates list to an array
params = np.array(params)

#Draw a polygon with the coordinates
cv2.drawContours(image,[params],-1 ,(255,0,150),-1)

cv2.imshow('no_conversion',image)

if(len(params)>=4):
    #Sort the coordinates
    params = order_coordinates(params)

#Draw the polygon with the sorted coordinates
cv2.drawContours(image,[params],-1 ,(255,0,150),-1)

cv2.imshow('detection',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```