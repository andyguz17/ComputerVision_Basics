import numpy as np
from skimage import exposure 
from skimage import feature
import cv2
 
img = cv2.imread('test_e.jpg')

imX = 720
imY = 1080
img = cv2.resize(img,(imX,imY))

#Here we define the main parameters of the model
#The number of orientations will be 9 as we explained before
#We will have 64 pixels per cell
#Now we can divide the blocks in cells of 16 * 16, so we define blocks as 2 by 2 
#each one of 8*8 pixels
(H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
	visualize=True)

hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

cv2.imshow('features',hogImage)

cv2.waitKey(0)
cv2.destroyAllWindows()

