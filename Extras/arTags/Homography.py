import cv2 
import numpy as np 
from cv2 import aruco

def order_coordinates(pts, var):
    coordinates = np.zeros((4,2),dtype="int")

    if(var):
        s = pts.sum(axis=1)
        coordinates[0] = pts[np.argmin(s)]
        coordinates[3] = pts[np.argmax(s)] 

        diff = np.diff(pts, axis=1)
        coordinates[1] = pts[np.argmin(diff)]
        coordinates[2] = pts[np.argmax(diff)]
    
    else:
        s = pts.sum(axis=1)
        coordinates[0] = pts[np.argmin(s)]
        coordinates[2] = pts[np.argmax(s)] 

        diff = np.diff(pts, axis=1)
        coordinates[1] = pts[np.argmin(diff)]
        coordinates[3] = pts[np.argmax(diff)]
    
    return coordinates

image = cv2.imread('a1.jpg')
h, w = image.shape[:2]

image = cv2.resize(image,(int(w*0.7), int(h*0.7)))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Initialize the aruco Dictionary and its parameters 
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

#Detect the corners and ids in the images 
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

#Initialize an empty list for the coordinates 
params = []

for i in range(len(ids)):

    #Catch the corners of each tag
    c = corners[i][0]

    #Draw a circle in the center of each detection
    cv2.circle(image,(int(c[:, 0].mean()), int(c[:, 1].mean())), 3, (255,255,0), -1)
    
    #Save the coordinates of the center of each tag
    params.append((int(c[:, 0].mean()), int(c[:, 1].mean())))

params = np.array(params)

if(len(params)>=4):

    #This function will sort the coordinates, so we can draw the polygon correctly
    params = order_coordinates(params,False)
    params_2 = order_coordinates(params,True)

paint = cv2.imread('paint_3.jpg')
height, width = paint.shape[:2]

coordinates = np.array([[0,0],[width,0],[0,height],[width,height]])

# Calculate Homography
hom, status = cv2.findHomography(coordinates, params_2)
  
# Warp source image to destination based on homography
warped_image = cv2.warpPerspective(paint, hom, (int(w*0.7), int(h*0.7)))

mask = np.zeros([int(h*0.7), int(w*0.7),3], dtype=np.uint8)

# Prepare a mask representing region to copy from the warped image into the original frame.
cv2.fillConvexPoly(mask, np.int32([params]), (255, 255, 255), cv2.LINE_AA)

substraction = cv2.subtract(image,mask)
addition = cv2.add(warped_image,substraction)

#Draw an square with the coordinate of the tags 
cv2.drawContours(image,[params],-1 ,(255,0,150),-1)

cv2.imshow('detection',addition)
cv2.waitKey(0)
cv2.destroyAllWindows()