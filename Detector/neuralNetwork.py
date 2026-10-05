# using a convolutional neural network to classify images as real or fake
# this will hopefully achieve a higher accuracy than the logistic regression model

# This first part is me learning about the tensorflow keras library and how to use it via the tensorflow documentation itself
# https://www.tensorflow.org/tutorials/images/cnn

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


