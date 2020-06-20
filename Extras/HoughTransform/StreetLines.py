import cv2 
import math
import numpy as np

def sort_param(x1, y1, x2, y2):
    empty_glass = 0

    if (y2 > y1):
        return x1,y1,x2,y2
    else:
        return x2,y2,x1,y1

image = cv2.imread('cde.jpg')
h,w = image.shape[:2]

#You will want to change the scale depending on your application 
#A lower resolution implies a higher speed, but less information to work with
scale = .5

image = cv2.resize(image,(int(w*scale),int(h*scale)))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#We define 45 degrees filter to catch som specific lines with convolution 
kernel_45 = np.array([[0,1,0],[1,0,-1],[0,-1,0]])
kernel_135 = np.array([[0,1,0],[-1,0,1],[0,-1,0]])

conv_45 = cv2.filter2D(gray,-1,kernel_45)
conv_135 = cv2.filter2D(gray,-1,kernel_135)

#Add both components
addition = conv_45+conv_135

#Use non local means denoising algorithm to reduce the nbackground noise on the image
addition = cv2.fastNlMeansDenoising(addition,addition,h=10,templateWindowSize=3,searchWindowSize=10)

#Eliminate intermidiate points with thresholding, so you have only white and black
ret,addition = cv2.threshold(addition,35,255,cv2.THRESH_BINARY)

#Function parameters
minLineLength = 350
maxLineGap = 100
theta = np.pi/180
rho = 1
threshold = 80

#Evaluate the image using the Hough transform to catch continous lines
lines = cv2.HoughLinesP(addition,rho,theta,threshold,minLineLength,maxLineGap)

#initialize empty lists to store the coordinates of the desired lines
left_line = []
right_line = []

try:
    for i in range(len(lines)):

        aux_l = 0
        aux_r = 0

        for x1,y1,x2,y2 in lines[i]:

            #First we need to sort our lines
            x1,y1,x2,y2 = sort_param(x1,y1,x2,y2)
            
            #Remove horizontal lines 
            if(abs(y1-y2)<=10):
                continue 

            #Remove 90° vertical lines
            if(abs(x1-x2)<=30):
                continue 
            
            #Calculate width and height
            y_diff = y2 - y1 
            x_diff = abs(x2-x1)

            #Orientation   
            #With this part we will divide between the left and right line
            #And save its values, also we will calculate the slope for each line             
            if(x2<x1):
                if(y_diff>aux_l):
                    aux_l = y_diff
                    left_line = [x1,y1,x2,y2]
                    m_left = (y_diff/x_diff)
                    x_left = x1 
                    y_left = y1
                else:
                    continue

            else: 
                if(y_diff>aux_r):
                    aux_r = y_diff           
                    right_line = [x1,y1,x2,y2]     
                    m_right = (y_diff/x_diff)
                    x_right = x1 
                    y_right = y1
                else: 
                    continue
    
    #This are just the projections of the lines, using basic calculus for a linear function
    #Line projection for the Left side 
    yl_0 = int(h*scale)
    xl_0 = int(((y_left-yl_0)/m_left)+x_left) 

    if(xl_0<=0):
        xl_0 = 0
        yl_0 = int(m_left*(-xl_0+x_left)+y_left)

    yl_1 = yl_0-int(yl_0*0.4)
    xl_1 = int(((y_left-yl_1)/m_left)+x_left)

    #Line projection for the right side
    yr_0 = int(h*scale)
    xr_0 = int(((-y_right+yr_0)/m_right)+x_right) 

    if(xr_0 > int(w*scale)):
        xr_0 = int(w*scale)
        yr_0 = int(m_right*(xr_0-x_right)+y_right)

    yr_1 = yl_0-int(yl_0*0.4)
    xr_1 = int(((-y_right+yr_1)/m_right)+x_right)

    #We draw the projection of the line 
    cv2.line(image,(xl_1, yl_1),(xl_0, yl_0),(255,0,255),8)
    cv2.line(image,(xr_0, yr_0),(xr_1, yr_1),(255,0,255),8)

except: 
    pass

cv2.imshow('result',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
