import cv2
import numpy as np

img = cv2.imread('example.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#The first step should be to get an edges image 
edges = cv2.Canny(gray,50,150,apertureSize = 3)

#Here we will define theparameteres for the detector 
#The theta represents the orientation accuracy which is 1°
theta = np.pi/180

#The rho represents the rows accuracy in pixels 
rho = 1

#Finally the threshold is the minimum number that the counter must have in order
#to be considered as a line (in other words could be described as the minimum number of 
#continous points)
threshold = 100

#Run and find the best candidates to be a line 
lines = cv2.HoughLines(edges,rho,theta,threshold)

#The lines vector will have the values of rho and theta 
#which counter is equal or greater than the threshold
images = []
for i in range(len(lines)):
    for rho,theta in lines[i]:

        #Using this information we can reconstruct the lines 
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        #draw the representing line in the original image
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow('drawn',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.waitKey(0)
cv2.destroyAllWindows()
