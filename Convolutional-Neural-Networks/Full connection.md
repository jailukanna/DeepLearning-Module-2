# Full connection


## Brief working of ANN/DNN

*A deep neural network (DNN) is an artificial neural network (ANN) with multiple layers between the input and output layers. The DNN finds the correct mathematical manipulation to turn the input into the output, whether it be a linear relationship or a non-linear relationship. The network moves through the layers calculating the probability of each output.*

*At first, the DNN creates a map of virtual neurons and assigns random numerical values, or "weights", to connections between them. The weights and inputs are multiplied and return an output between 0 and 1. If the network did not accurately recognize a particular pattern, an algorithm would adjust the weights.*

![dnn1.jpg](images/dnn1.jpg)

*The DNN classifier consists of neuron layers, which work using the activation function, called rectified linear unit (ReLU). The mathematical form of the DNN using SCMR-SVM  input is given in Eq. (1). The output layer relies on the cost function and softmax function, which is regarded as cross-entropy.\

The activation function rectifier is thus given as follows:\

f(x)=max(0,x) (1)\
where x is neuron input from FSVM, which is a ramp function.\

The rectifier unit ReLU provides a smooth approximation to a rectifier using an analytic function, which is given as:\

f(x)=ln[1+exp(x)]  (2)\
This is considered a softplus function.\

Once the prediction is carried out, a new raw descriptor representation is thus extracted from the hidden layers, which is given as:\

Xl+1=H(WlXl+Bl)(3)\
where H is the activation function, Wl is the weight matrix, and Bl is the bias of lth hidden layer and the selection of these parameters is based on the rectified linear unit (ReLu). Here, the input for the hidden unit is not directly supplied by the input training or testing dataset, but is handled by the FSVM.*



## Concept of Backpropagation

### What is Backpropagation

**Back-propagation is the essence of neural net training. It is the method of fine-tuning the weights of a neural net based on the error rate obtained in the previous epoch (i.e., iteration). Proper tuning of the weights allows you to reduce error rates and to make the model reliable by increasing its generalization.\
Backpropagation is a short form for "backward propagation of errors." It is a standard method of training artificial neural networks. This method helps to calculate the gradient of a loss function with respects to all the weights in the network.**

## How Backpropagation Works: Simple Algorithm
**Consider the following diagram**

![b1.png](images/b1.png)

**1.Inputs X, arrive through the preconnected path\
2.Input is modeled using real weights W. The weights are usually randomly selected.\
3.Calculate the output for every neuron from the input layer, to the hidden layers, to the output layer.\
4.Calculate the error in the outputs\
ErrorB= Actual Output – Desired Output\
5.Travel back from the output layer to the hidden layer to adjust the weights such that the error is decreased.\
Keep repeating the process until the desired output is achieved**

### Why We Need Backpropagation?
**Backpropagation is fast, simple and easy to program\
It has no parameters to tune apart from the numbers of input\
It is a flexible method as it does not require prior knowledge about the network\
It is a standard method that generally works well\
It does not need any special mention of the features of the function to be learned**

### Types of Backpropagation Networks
*Two Types of Backpropagation Networks are:\
1.Static Back-propagation\
2.Recurrent Backpropagation*

### Static back-propagation:
***It is one kind of backpropagation network which produces a mapping of a static input for static output. It is useful to solve static classification issues like optical character recognitio***

### Recurrent Backpropagation:
***Recurrent backpropagation is fed forward until a fixed value is achieved. After that, the error is computed and propagated backward.***\
*The main difference between both of these methods is: that the mapping is rapid in static back-propagation while it is nonstatic in recurrent backpropagation.*

## Full working fo the entire connection-CNN+ANN


*The general architecture of these combinations is a convolutional feature extractor applied on the input, then some recurrent network on top of the CNN’s output, then an optional fully connected layer on RNN’s output and finally a softmax layer.*

*The output of the CNN is a set of several channels (also known as feature maps). We can have separate GRUs acting on each channel (with or without weight sharing) as described in this picture:*

![r1.png](images/r1.png)

*Another option is to interpret CNN’s output as a 3D-tensor and run a single GRU on 2D slices of that tensor:*

![r2.png](images/r2.png)

*The latter option has more parameters, but the information from different channels is mixed inside the GRU, and it seems to improve performance. This architecture is similar to the one described in this paper on speech recognition, except that they also use some residual connections (“shortcuts”) from input to RNN and from CNN to fully connected layers. It is interesting to note that recently it was shown that similar architectures work well for text classification*

![Capture.PNG](images/Capture.PNG)


```python

```
