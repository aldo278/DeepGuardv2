## DeepGuard v2

This is the second iteration of the DeepGuard project, which aims to detect AI-generated images using various machine learning techniques.
Unfortunately, video data and metadata is around 16 GB of data, and since my laptop sucks and I don't think lab pc's will be able to handle that amount of data, this project will only focus on images.

I will be using the following dataset from kaggle:
- CIFAKE: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

The last version used an API that i tweaked so that it took 5 frames from a video, where it eventually correctly predicted the fake video , so I might be able to implement something similar.

I will be giving updates on my progress in this file, and will be adding new sections as I go along. Newest updates will be at the top.

---------------------------------------------------------------------------------


## Detector
Before using a model, I decided to first check if The given image contains a hidden watermark, as this would be a good indicator of whether the image is real or fake.

watermark.py was originally named data.py but i decided to use the same file to detect a watermark

The training data images are too small thus the watermark detector library cannot make out the image, let alone check for a hidden watermark. Skip this for now, I will try to add a loop where i give a UI fake images and annotate them myself. Moving on to Logistic regression...


### Logistic Regression
The current model has a 68% accuracy, its okay for being the first take but not ready as MVP. 
Another thing to note is that the script logisticRegression.py takes around 5 min to run and return results, it might be because of the large size of the dataset.
Sigmoid function: activation function that squashes the output between 0 and 1, making it suitable for binary classification.


### Convolutional Neural Network (CNN)
I will try to implement a CNN model with the current dataset, and see if it can achieve a higher accuracy.
If it fails, I will try to get my own data. Either way making my own dataset with current images would help not only on accuracy but also on the watermark detection.


I did some research and found that CNNs are implement with either TensorFlow or PyTorch. I will try to use TensorFlow first, as it is more beginner-friendly.

Definitions:
Tensor: a tensor is a container for numbers arranged along some number of dimensions. It is the basic data structure in TensorFlow, and its essentially the same idea as a numpy array.
with extra features for running on GPU's and tracking gradients. So a 3d matrix would be a 3d tensor.

Convolution: a mathematical operation that slides a small grid of numbers, called a kernel (or filter), over an input array (such as an image or time-series data) to 
extract local patterns and spatial relationships.
What the convolution does, step by step:
- take a small filter (a grid of weights, here 3x3)
- place it over a 3x3 patch of the image
- multiply each weight by the pixel under it, then add everything up, that gives one number
- slide the filter over and repeat, across the whole image.
The numbers collected from all positions form one feature map: a new image showing where that filter pattern was found


Convolutional Base: The first part of a CNN, which is a stack of Conv2D and MaxPooling2D layers that extract features from the image. In this project it can find differences between neighboring pixels. (face isolation is a separate step). Some features include edges, gradients, color contrasts, and simple textures, middle layers can detect more complex patterns like corners, repeated patterns, skin, or fur or fabric textures. Deeper layers get larger, task specific features like facial features, objects, or text.

Rectified Linear Unit (ReLU): an activation function in neural networks that outputs the input directly if it is positve, and outputs zero if the input is negative or zero
In a Con2D layer, this is applied to every value in every feature map, right after the filter's weighted sum. It is an activation function that introduces non-linearity into the model, allowing it to learn more complex patterns. Kind of like the sigmoid function.

For the CNN, there are algorithms that help in adjust weights to minimize error.
- Adam optimizer: a popular optimization algorithm that adapts the learning rate for each parameter based on estimates of the first and second moments of the gradients
- sgd (stochastic gradient descent): uses one fixed step size for everything, whereas Adam's default learning rate is much smaller (0.001)


will it differeciate between CGI and AI?
check if too smooth or too irregular for CGI
sobel operators
https://en.wikipedia.org/wiki/Sobel_operator

What is an Epoch????
An Epoch is one full pass through the entire training set. With 50,000 training images, one epoch means the model has seen all 50,000 once (in batches of 32, so about 1563 weight updates). In the first NN, we set epochs=10m so it made 10 passes.

Why more than one pass? A single look at the data isn't enough for the weights to settle on good values. Each pass improves them a bit more, like re-reading a textbook chapter several times. Too many epochs, though, ans the model starts memorizing the training images instead of learning general patterns (overfitting), which is exactly what the plot helps spot.

![logo](https://365datascience.com/resources/blog/thumb@1024_bpm7kc63z8-overfitting-vs-underfitting-models-examples.webp)
