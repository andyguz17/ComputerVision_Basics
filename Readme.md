#  Computer Vision Basics

## Introduction 

In this course we will talk about the basics of computer vision, in which we will emphasize the following topics:

- Color and color filtering.
- Morphologic filters. 
- Edge detection and a brief introduction to convolutions. 
- Color Spaces.

### What Compuiter Vision is ? 

Computer vision commonly abbreviated as CV could be described as a field of study that allows a computer to analise and have understanding of an uploaded digital image or group of images such as videos. 

The main idea of ​​CV along with robotics and other fields of study is to help and improve tasks that could be exhaustive or repetitive for humans. In recent years, there have been many improvements with the invention of complex systems of computer vision and deep learning, such as the well-known convolutional neural networks, which changed the point of view to solve many problems, such as facial recognition or medical images among others. 

For this course in specific we are going to make use of python 3.5 and opencv, despite this python version can be considered a little old, is a very stable version, however you can fell free to change to newest versions like python 3.7, some featueres may change, but it keeps the main idea. 


### Images

First of all we need to understand what exactly an image, colloquially we could describe it as a visual representation of something that itself is a set of many characteristics as color, shapes, etc. For a computer an image could be better described as a matrix, in which every value is considered a pixel, so when your a talking about a 1080p image resolution, you´re refering to an specific 1080*1920 px matrix.

#### Color 

In the case of an colored imaged, we are talking about a thre-dimmensional matrix where each dimension corresponds to an specific color chanel (Red, green, blue), the dimensions of this matrix will be different for different color spaces which we will discuss further in the course. 

We can describe an image in many more complex ways, like the color construction that is a result mainly of the light over the object surface, when we have something black it is actually the lack of light, the color formation will depend on the wavelength of the main components of the white light, the infrared and ultraviolet rays. 

If you like physics as much as I you will find an interesting phenomenom where the color deformation can be seen, the stars, you can see in many pictures of the space that the rock formations that are way too far from us has a red color while the closest ones has a blue color, this phenomenom was discovered by the North American Astronomer Edwin Hubble in 1929, we know that the space is in constant expantion, so if the sapce is deformed, the light that we receive from that stars will suffer from that expansion too, in consequence the wavelenght of the light will be higher and the color we will perciebe will have a red tone instead of a blue one for example. 

I don´t want to go much deeper on the color formations and theory of it, the main idea is so you can know what are we going to work with for the rest of the course, anyway it will be helpfull if you want to do a more profound research on this topics, that I consider are really interesting. 

