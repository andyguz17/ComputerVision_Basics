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
