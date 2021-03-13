# Convolutional Neural Networks

## Convolution Operations:

### What is a Convolution?
*A convolution is how the input is modified by a filter. In convolutional networks, multiple filters are taken to slice through the image and map them one by one and learn different portions of an input image. Imagine a small filter sliding left to right across the image from top to bottom and that moving filter is looking for, say, a dark edge. Each time a match is found, it is mapped out onto an output image*


*Convolutional Operation means for a given input we re-estimate it as the weighted average of all the inputs around it. We have some weights assigned to the neighbor values and we take the weighted sum of the neighbor values to estimate the value of the current input/pixel*

## what are Filters


1.*Filter is referred to as a set of shared weights on the input*\
2.*Filters detect spatial patterns such as edges in an image by detecting the changes in intensity values of the image*\
3.*A filter is represented by a vector of weights with which we convolve the input. The filter, similar to a filter encountered    in signal processing, provides a measure for how close a patch of input resembles a feature. A feature may be vertical edge      or an arch.The feature that the filter helps identify is not engineered manually but derived from the data through the          learning algorithm*

## What is convolution Layer 

*Convolution is the first layer to extract features from an input image. Convolution preserves the relationship between pixels by learning image features using small squares of input data. It is a mathematical operation that takes two inputs such as image matrix and a filter or kernel.*

![im1.png](images/im1.png)

*Consider a 5 x 5 whose image pixel values are 0, 1 and filter matrix 3 x 3 as shown in below*

![im2.png](images/im2.png)

*Then the convolution of 5 x 5 image matrix multiplies with 3 x 3 filter matrix which is called “Feature Map” as output shown in below*

![im3.gif](images/im3.gif)

*Convolution of an image with different filters can perform operations such as edge detection, blur and sharpen by applying filters. The below example shows various convolution image after applying different types of filters (Kernels).*

![im4.png](images/im4.png)

## Need and working of convolution layer

*CNNs compare images piece by piece. The pieces that it looks for are called features. By finding rough feature matches in roughly the same positions in two images, CNNs get a lot better at seeing similarity than whole-image matching schemes*\
*Each feature is like a mini-image a small two-dimensional array of values. Features match common aspects of the images. In the case of X images, features consisting of diagonal lines and a crossing capture all the important characteristics of most X’s. These features will probably match up to the arms and center of any image of an X*

![c1.png](images/c1.png)

*When presented with a new image, the CNN doesn’t know exactly where these features will match so it tries them everywhere, in every possible position. In calculating the match to a feature across the whole image, we make it a filter. The math we use to do this is called convolution, from which Convolutional Neural Networks take their name*\

*To calculate the match of a feature to a patch of the image, simply multiply each pixel in the feature by the value of the corresponding pixel in the image. Then add up the answers and divide by the total number of pixels in the feature. If both pixels are white (a value of 1) then 1 * 1 = 1. If both are black, then (-1) * (-1) = 1. Either way, every matching pixel results in a 1. Similarly, any mismatch is a -1. If all the pixels in a feature match, then adding them up and dividing by the total number of pixels gives a 1. Similarly, if none of the pixels in a feature match the image patch, then the answer is a -1.*

![c2.png](images/c2.png)

*To complete our convolution, we repeat this process, lining up the feature with every possible image patch. We can take the answer from each convolution and make a new two-dimensional array from it, based on where in the image each patch is located. This map of matches is also a filtered version of our original image. It’s a map of where in the image the feature is found. Values close to 1 show strong matches, values close to -1 show strong matches for the photographic negative of our feature, and values near zero show no match of any sort.*

![c3.png](images/c3.png)

*The next step is to repeat the convolution process in its entirety for each of the other features. The result is a set of filtered images, one for each of our filters. It’s convenient to think of this whole collection of convolution operations as a single processing step. In CNNs this is referred to as a convolution layer,*

## ReLU Layer

*The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. It has become the default activation function for many types of neural networks because a model that uses it is easier to train and often achieves better performance.*

*In order to use stochastic gradient descent with backpropagation of errors to train deep neural networks, an activation function is needed that looks and acts like a linear function, but is, in fact, a nonlinear function allowing complex relationships in the data to be learned.*\
The function must also provide more sensitivity to the activation sum input and avoid easy saturation.\
The solution is to use the rectified linear activation function, or ReL for short.\
A node or unit that implements this activation function is referred to as a rectified linear activation unit, or ReLU for short. Often, networks that use the rectifier function for the hidden layers are referred to as rectified networks.*

*The rectified linear activation function is a simple calculation that returns the value provided as input directly, or the value 0.0 if the input is 0.0 or less.\
We can describe this using a simple if-statement*

*if input > 0:\
	return input\
else:\
	return 0*

*We can describe this function g() mathematically using the max() function over the set of 0.0 and the input z; for example*

*g(z) = max{0, z}*

*The function is linear for values greater than zero, meaning it has a lot of the desirable properties of a linear activation function when training a neural network using backpropagation. Yet, it is a nonlinear function as negative values are always output as zero*

### Tips for Using the Rectified Linear Activation

*Use ReLU as the Default Activation Function\
Use ReLU with MLPs, CNNs, but Probably Not RNNs\
Use “He Weight Initialization”*

### Advantages of the Rectified Linear Activation Function

*1.Computational Simplicity\
 2.Representational Sparsity\
 3.Linear Behavior\
 4.Train Deep Networks*


```python

```
