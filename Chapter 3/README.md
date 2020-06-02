# Feature Matching 

Feature matching are a group of algorithms that play an important role in some computer vision applications. The main idea of these is to extract important features from a training image (That contains an specific object) then extract features from other images, where the desired object can be, we will want to compare the features from both images and match them if there is any similitud in between these two images, if there is many matching points, it can happen that the desired object is in the second image as well. 

<div style="text-align:center"><img src="Resources/Matching.png" width = 80% /></div>
<br>

A feature can be defined as a distinctive attribute or aspect of something, we want these features to be unique for each object, so we can recognize them every time we see them in other images. 

When we are talking about images a features is a piece of relevant information, these can be in specific locations, like some shapes, mountain peaks, corners, etc. These are called keypoints features, these are usually described by a patch of surrounding neighbour pixels. Also we can describe some features through it's edge profile, describing the local appearance and orientation.

Working with Feature matching is pretty straightforward, in general we need to follow some steps:

- Identify Points of Interest.
- Description of the point of interest, desccribing its surrounding, the idea of this step to make de algorithm robust to image transformations, like plane rotation, illumination changes, scale variation, etc. 
- Matching, where you want to compare the features of your object with the ones in other images, seaching for similarities between them.

### Features from Accelerated Segment Test (FAST)