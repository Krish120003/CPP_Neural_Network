# Neural Networks in C++

## Running

To run the project, run the following commands:

```bash
cmake .
make
./main
```

This project is a C++ implementation of a neural network. It is a simple fully-connected neural network that can be used to classify handwritten digits from the MNIST dataset. To change the network structure or experiment with other changes, change the `main.cpp` or other files in the `src` folder.

Dataset used: [MNIST](http://yann.lecun.com/exdb/mnist/)

## Report - Table of Contents

- [Neural Networks in C++](#neural-networks-in-c)
  - [Table of Contents](#table-of-contents)
  - [Preface](#preface)
  - [Project Introduction](#project-introduction)
  - [Method](#method)
    - [Using the Dataset](#using-the-dataset)
    - [Definitions](#definitions)
      - [The Neuron](#the-neuron)
      - [The DenseLayer](#the-denselayer)
      - [The Activation Layers](#the-activation-layers)
      - [Mean Squared Error as Loss](#mean-squared-error-as-loss)
    - [The Network](#the-network)
      - [Structure 1](#structure-1)
      - [Structure 2](#structure-2)
      - [Training](#training)
      - [Evaluation](#evaluation)
      - [Results](#results)
  - [Project Extension](#project-extension)
    - [A New Network Structure](#a-new-network-structure)
    - [Evaluating Various Activations Functions](#evaluating-various-activations-functions)
      - [The Sigmoid Activation Function](#the-sigmoid-activation-function)
    - [Important Note about Training](#important-note-about-training)
    - [Results](#results-1)
      - [Actual Results](#actual-results)
      - [Understanding the Results](#understanding-the-results)
        - [ReLU](#relu)
        - [LeakyReLU](#leakyrelu)
        - [Sigmoid](#sigmoid)
      - [Visualizing Example Outputs](#visualizing-example-outputs)
  - [Conclusion](#conclusion)
  - [References](#references)

## Preface

Neural networks are a computational model that try to simulate the human brain to solve tasks. Neural networks are made from a large amount of simple non-linear equations that can be used to solve complex problems. These complex problems often include classification, time-series predictions, image segmentation and many other computer vision and language prediction.

One of the most common databases used to test neural networks for classification is the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. The MNIST database is a set of handwritten digits that has a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 pixels greyscale image.

![](assets/Figure1.png)

$$
\text{Figure 1: A few examples from the MNIST dataset}
$$

## Project Introduction

For this project, we try to recreate a simple fully-connected neural network to classify an image of handwritten digits. To do this, we implement the following in C++:

- Data Loading, to load the binary files for the MNIST dataset into vectors.
- A Neuron class, to represent one Neuron in the neural network
- A DenseLayer class, to represent and create fully-connected layers of neurons
- A ReLuLayer class, to represent the ReLU activation function for neurons
- A MeanSquaredError class, to compute the error between the outputs of the network and the expected outputs.
- A training loop, which trains the network by back-propagating and adjusting the weights of each neuron in the network to make the network "learn" about the dataset
- A loop over the test data that the network did not learn from, to evaluate the performance of the network.

All of the above were implemented and shown in the in-class demo for this project.

## Method

### Using the Dataset

The MNIST dataset has 28x28 pixel images, and a single integer representing the digit for each image. We can represent all the labels for images as a vector of integers. For the images, we can "flatten" each 28x28 image into a vector of $28\cdot 28 = 784$ values. This will be the input to our network/each neuron.

### Definitions

#### The Neuron

Let $O$ denote the output of a single neuron in  
$O = f((w \cdot i) + b)$  
where

- $w$ indicates a vector containing the weight for each of the input
- $i$ indicates the input as a vector
- $b$ indicates a single number, a bias value
- $f$ is the activation function.

In our model, the input would be our vector of 784 values, the pixels of each image. The $w$ and $b$ are initialized randomly, and we choose $f$ to be the $ReLU$ activation function and the identity function for our initial demo. However, those are defined as separate layers for easier implementation.

To "train" our network, we would do a forward pass through our neurons - find a final output, compute the "loss", how different the output is from the expected value, and calcluate the gradient/derivative of each of the weight with respect to this loss. We want to minimize this loss, so we would descend on the gradient by adjusting our weights.

This would simplify to:

$$
newWeights = weights - w_{gradients}
$$

$$
newBias = bias - bias_{gradient}
$$

However, we don't want to drastically change weights and learn incrementally, so its common to multiply the gradients by a small number called the "learning rate". This is implemented as well, and we will show how it affects training later in results.

#### The DenseLayer

A dense layer, $L_i$ is a collection of neurons that feeds its output forward to the next layer $L_{i+1}$. DenseLayers can be chained together, or have different types of layers in between them. For our project, we only explore simple activation layers between dense layers.

#### The Activation Layers

For the demo, we implement:

- ReLU
  - Defined as $y = max(0, x)$
  - This is a non-linear activation, that prevents negative values.
  - Simulate either a neuron not being active at all, or firing with some strength $x$
- LeakyReLU
  - Defined as $y = max(x, x*LEAK)$ where $0 \lt LEAK \lt 1$
  - This is a non-linear activation function that decreases the magnitude of negative neuron outputs in the network.

![](assets/ReLuActivation.png)

$$
\text{Figure 2: The ReLU activation function.}
$$

![](assets/LeakyReLuActivation.png)

$$\text{Figure 3: The Leaky ReLU Activation, with LEAK=0.25}$$

Each of the activation layers is implemented as it's own class, and works similarly to DenseLayer. These classes are designed to make it easy to forward pass / back-propagate between each other.

#### Mean Squared Error as Loss

We need some metric to measure how "different" the output of our network is from our expected labels. For this, we utilize the mean squared error loss, defined as the following:

$$
MSE = \dfrac{1}{N}\sum_{i=1}^n (Y_i - \hat{Y_i})^2
$$

The MSE allows us to measure how different our output is from the expected, even across a vector. Since we are doing categorization, we can use a vector of size 10 to represent the probability of a certain output being a label. MSE allows us to compute the difference across this vector.

Our goal is to minimize the loss; to minimize the squared difference between our outputs.

### The Network

#### Structure 1

For the demo, we compose the following layers to create a network:

- $D_1$ - A DenseLayer with $784$ inputs and $100$ outputs.
- $A_1$ - A ReLU Activation Layer
- $D_2$ - A DenseLayer with $100$ inputs and $1$ output.
- $A_2$ - A ReLU Activation Layer
- $Mean Squared Error$ Loss

#### Structure 2

For the demo, we compose the following layers to create a network:

- $D_1$ - A DenseLayer with $784$ inputs and $100$ outputs.
- $A_1$ - A LeakyReLU Activation Layer
- $D_2$ - A DenseLayer with $100$ inputs and $1$ output.
- $A_2$ - A LeakyReLU Activation Layer
- $Mean Squared Error$ Loss

#### Training

An $epoch$ stands for one iteration of training over the entire training dataset.  
We train these model for $50$ epochs, in the following manner:

Define the $LearningRate = 0.0001$

For every image defined as a vector of 784 doubles in the training set:

- Feed forward the vector into $D_1$, getting back an output of $100$ doubles
- $A_1$ - Apply the ReLU activation function element-wise to the output of $D_1$, resulting in a vector of $100$ doubles.
- $D_2$ - Feed forward the ReLU-activated vector into $D_2$, obtaining an output of $10$ doubles.
- $A_2$ - Apply the ReLU activation function element-wise to the output of $D_2$, resulting in a vector of $10$ doubles.
- Loss Calculation - Compute the Mean Squared Error (MSE) loss between the output of $A_2$ and the true label int (ground truth).
- Set gradients of all weights and biases in all layers to be 0.
- Set the initial gradient of the loss layer to 1.
- Backward propagate through the network, computing the gradients of each weight and bias with respect to the final loss.
- Use the computed gradients to _descend_ to a lower loss value. We do this by updating the each weight $w$ to have value $w = w - w_{grad} * LearningRate$ and each bias $b$ the same way.

We do this for both network structure 1 and 2 (but with LeakyReLu layers for structure 2).

#### Evaluation

After each epoch, we want to check the accuracy of the network to see how it is doing. In this case, we are outputting one single number. We can round this number and convert it to an integer, and refer to it as $Y_i$.

For each output $Y_i$ on the training dataset, we can compute a vector $Y$, predictions for labels of all images in the training set. Then, we can simply check if each $Y_i = \hat{Y_i}$ and find the **accuracy** of the network.

Additionally, we keep track of the $mean$ value of the loss for each Epoch to see how the network learns.

![](assets/Figure4.png)

$$
\text{Figure 4: Loss Over Epocs/Training Time}
$$

![](assets/Figure5.png)

$$
\text{Figure 5: Accuracy Over Epochs/Training Time}
$$

#### Results

From the above, we can see that:

- The ReLU network has a high loss, and almost 10% accuracy.
- The LeakyReLU network has a much lower loss, and is getting better but very very slowly.

By investigating some outputs, we would find that the $ReLU$ network finds a _local minima_ in its gradients - the outputs of the neurons are negative, and thus the network only ever outputs 0. This leads to a $10\%$ accuracy, which is as expected since about 10% of the labels in the dataset are 0. This is a common problem in neural networks called [The Dying ReLU Problem](https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks). When values reach 0, the gradients are all 0, and thus the network can never learn or recover from this.

A fix for this problem is the LeakyReLu, where negative values still exist instead of just 0 to let the network still learn. We see above that this is much better, as the network learns some values and has an accuracy of about $20$% and increasing. However, the network is learning very very slowly, so we stop training here (especially since an unoptimized C++ implementation takes a while to run every epoch.)

From this demo, we can conclude that we have achieved some learning with the above network and the LeakyReLU activation, but we would need to train the network significantly longer for it to learn to a reasonably good accuracy.

Since we have a working set of classes such as Neurons, DenseLayers, MeanSquaredError Loss and some activation functions, the next steps would be to experiment with the structure of the network to improve its performance.

## Project Extension

While we implemented the basics of a network, there are many things to explore to further improve this neural network. In this extension, we show how differentiating activation functions on neurons and a different structure can lead to a variety of improvements on the network performance.

### A New Network Structure

In the demo, our network output a single `double` value as the predicted label for an image. However, our goal here is classification - to classify each image to be 1 of 10 digits. Instead of outputting 1 number, we can change our network to output 10 different values. Each of these values will represent what the network thinks is the likely label for the image. We also change our $\hat{Y}$ values to be [one-hot encoded vectors](https://www.educative.io/blog/one-hot-encoding). This means instead of representing a label as a single int, we represent it as a vector of 0's, with the value at the index of the label being set to 1. For example, we encode $5$ as:

$$
[0,0,0,0,0,1,0,0,0,0]
$$

where the $5th$ index of the vector is 1, and the rest are 0.

Thus, our new network structure would be:

- $D_1$ - A DenseLayer with $784$ inputs and $100$ outputs.
- $A_1$ - An Activation Layer
- $D_2$ - A DenseLayer with $100$ inputs and $10$ outputs.
- $A_2$ - An Activation Layer
- $Mean Squared Error$ Loss

We leave $A_1$ and $A_2$ be unspecified activation functions, and treat them as our experimental variables.

### Evaluating Various Activations Functions

#### The Sigmoid Activation Function

The [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function is defined as:

$$
\sigma(x) = \dfrac{1}{1+e^{-x}}
$$

![](assets/Figure6.png)

$$
\text{Figure 6: The Sigmoid Function}
$$

It is often referred to as a squashing function, squashing any value of $x$ between 0 and 1. We implemented this in the source code as the SigmoidLayer, implementing the forward pass and backward pass with equations as defined in the [Sigmoid wikipedia page.](https://en.wikipedia.org/wiki/Sigmoid_function)

### Important Note about Training

When training the network with ReLU and LeakyReLU, the learning rate was set to be a small value equal to $0.0001$. With a larger learning rate, each gradient will have more effect on each weight each step. However, if we increased the learning rate in an attempt to train the network faster, we would notice the following:

```
Epoch 1:
i:0   | Mean Loss: 3646
i:500 | Mean Loss: -nan
```

Here, the weights in the network grow really really fast, thus causing the outputs of the network to overflow C++ doubles and create `nan` values. Once part of the network starts outputting `nan`, the network is unable to recover from this. Thus, we choose a very small learning rate.

However, when training with the Sigmoid activation, a learning rate of $0.0001$ is very small, and it takes a large amount of steps for the network to learn anything since the outputs are always squashed between 0 and 1. For example, with a learning rate of $0.0001$, we get the following when training with Sigmoid:

```
Epoch: 0 | Loss: 1.79 | Train Accuracy: 0.1022
Epoch: 1 | Loss: 1.738 | Train Accuracy: 0.1022
Epoch: 2 | Loss: 1.721 | Train Accuracy: 0.1068
```

We can see here that even after 3 epochs, the network is barely learning because the weights change very very slowly. Efficient network training means that the network learns at a reasonably good pace, thus we increase the learning rate to $0.1$.

### Results

Now, we will evaluate the following activation functions with our new network structure.

- ReLu
- LeakyReLu with Leak = 0.25
- Sigmoid

We use the following metrics to measure our network:

- Training Loss
- Training Accuracy
- Test Accuracy

The test accuracy refers to accuracy on a dataset that the network has never seen/learned from before.

#### Actual Results

![](assets/Figure7.png)

$$
\text{Figure 7: Loss Over Epoch/Training Time}
$$

![](assets/Figure8.png)

$$
\text{Figure 8: Accuracy Over Epoch/Training Time}
$$

Additionally, we find the following final accuracy over the test data (this is data the network has not learned from. It helps us see if the learnings generalize to other data.)

- ReLU: 9.8%
- LeakyReLU: 87.66%
- Sigmoid: 90.42%

#### Understanding the Results

From the results above, we can see that this new network structure enables LeakyReLU to successfully learn and classify the MNIST handwritten digits compared to our previous structure. We also see that the new Sigmoid activation function performs very well in learning and classifying the digits as well. However, the ReLU function struggles and stays at a loss of 1.0 and an accuracy of about 10%.

##### ReLU

Firstly, we see that the ReLU network maintains a loss of 1.0 and an accuracy of 10%. Why? Well this is because the network finds a local minima, where every output vector is:

$$
[0,0,0,\dots,0,0]
$$

This allows the network to consistently achieve a loss of 1, since we expect one of the labels to be 1 and the rest to be 0, and the network always outputs 0s, so its always close to the output (but not correct).

##### LeakyReLU

Secondly, we see that the LeakyReLU activation is able to learn the dataset much better than with the previous setup. This is likely because now the network is trying to minimize all 0-values, and maximize the active label. It cannot have a dying-network problem like the ReLU network, so the network slowly learns the dataset correctly over a large number of epochs. As explained earlier, the learning rate has to be a small number to prevent overflow, so we see the network accuracy get better slowly over time.

##### Sigmoid

Finally, we see that the Sigmoid activation excels at learning our dataset. It gets to 80% accuracy in less than 5 epochs, and continuously learns over time. This is likely because the network outputs are always squashed between 0 and 1, leading to a much more stable network. Additionally, because the network is stable, the higher learning rate means the network learns much much faster, as shown in the graphs above.

#### Visualizing Example Outputs

Let's examine some outputs of this network! Given a random input image, we see the following:

**LeakyReLU**

```

                 ,$,
                .$$
                #$,
               O$#
              .$$.
              #$O
             O$#
            .$$:
            O$#
           .$$o
           O@#
           X$o
           $$oo#o
          o$$$$$$o
          O$$$O,$O
          #$$X .$X
          X$$o :$X
          :$$O:$$:
           X$$$@#
            o##X.



Prediction: 6 | Label: 6
Probabilities: [-0.01168, -0.006693, 0.01363, -0.007407, -0.00899, -0.009166, 0.496, -0.01396, -0.004971, -0.01538]
```

![](assets/Figure9.jpeg)

$$
\text{Figure 9: Sample Network Output for LeakyReLU}
$$

**Sigmoid**

```
Prediction: 6 | Label: 6
Probabilities: [3.626e-05, 0.008488, 0.01929, 0.01381, 0.0001375, 0.002866, 0.5032, 0.002111, 0.04593, 0.001489]
```

![](assets/Figure10.jpeg)

$$
\text{Figure 10: Sample Network Output for Sigmoid}
$$

From these outputs, we can see that both networks have all values except the correct one close to 0. In Sigmoid, some values are very very small, whereas LeakyReLU has some negative values.

Let's visualize another value:

```




       oOX@$@$$$o
      O$$$$XO#$$$X
     O$o,       X$#.
     O$          #$#
     .,           #$X
                  :$$,
                   X$$
                   o$$,
                   .$$:
          :O$$$$XO, $$:
        ,#$$$$$$$$$#$$
        $$O::.. :o$$$$
       O$X        ,$$$
       X$X        .$$O
       X$X        #$$#
       X$X       #$$.
       :$$o    .$$$:
        X$$O,,O$$$:
        .X$$$$$$$:
          :X$$$o.




Sigmoid Prediction: 3 | Label: 2
LeakyReLU Prediction: 6 | Label: 2
```

![](assets/Figure11.jpeg)

$$
\text{Figure 11: Sample Network Output for LeakyReLu and Sigmoid on incorrect prediction.}
$$

From the above graph, it's interesting to see how wildly different results both activations achieve starting from the exact same weights and on the exact same inputs. Sigmoid indicates that this is likely a 2 or 3, whereas LeakyReLU is all over the place.

## Conclusion

In conclusion, this project explored how to build neural networks from scratch, implementing forward pass and back-propogation, and making networks learn. In the extension, we learned that different network structured can lead to very different results in whether the network is able to learn the data or not. We also explored how different functions used as activations for neurons can cause the network to learn differently, and how some functions such as Sigmoid have stable mathematical properties that make network training much faster and more accurate.

## References

https://www.bargraphmaker.net/  
https://en.wikipedia.org/wiki/Neural_network  
https://nnfs.io  
https://cs50.harvard.edu/ai/2020/notes/5/  
https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi  
https://en.wikipedia.org/wiki/Sigmoid_function  
https://en.wikipedia.org/wiki/Rectifier_(neural_networks)  
https://en.wikipedia.org/wiki/Mean_squared_error  
https://www.youtube.com/watch?v=aircAruvnKk  
https://www.desmos.com/calculator
