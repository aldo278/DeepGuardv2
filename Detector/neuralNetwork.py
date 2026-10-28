# using a convolutional neural network to classify images as real or fake
# this will hopefully achieve a higher accuracy than the logistic regression model

# This first part is me learning about the tensorflow keras library and how to use it via the tensorflow documentation itself
# https://www.tensorflow.org/tutorials/images/cnn

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt


# this is python's tuple unpacking, and we do it like this because load_data() returns a nested structure tuples
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalize pixel values to be between 0 and 1
# pixel values in an image are stored as integers from 0-255, where 0 is dark and 255 is bright
# we divide by 255 to rescale that range to 0-1 as a float, getting decimals instead of just 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
# Training becomes more stable so that weights start at small values and keep everything in a manageable range


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # The CIFAR lables are arrays
    # which is why you need an extra index
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


import tensorflow as tf
# now we create a convolutional base:
# The convolutional base is the first part of a CNN: the stack of Conv2D and MaxPooling2D layers
# that extract features from the image, not a special keras object, just a name for that section of the model

#first we create an emty model where layers are stack in a straight line, with each layer's output feeding into the next
# Sequential works when the data flows throu one path with not branches
model = models.Sequential()

# here we add a 2d convolutional layer: it slides small filters acroos an image to detect patterns, it is the core building block of a CNN
# we specify the number of filters (32), their size (3x3), and the activation function (relu), input shape is the heigh, width, channels of one image (RGB) only needed in the first layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# here we add another layer that downsamples. it slides a 2x2 window over each feature map, moving two pixels at a time so there is no overlap
# and takes the maximum value in that window, reducing the spatial dimensions by half
model.add(layers.MaxPooling2D((2, 2)))

# another layer with more filters shrinks the image further while extracting more complex patterns
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 13x13 becomes a 6x6 
model.add(layers.MaxPooling2D((2, 2)))

# another layer with more filters 
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# The output of every Con2D and MaxPooling2D layer is a 3D tensor of shape (height, width, channels). The width and height dimensions tend to shrink as we go deeper in the network
# The number of output channels for each Conv2D layer is controlled by the firs argument. Typically, as the width and height shrink, you can afford (computationally) 
# to add more output channels in each Conv2D layer

# now we can add Dense layers on top
# To complete the model, we can feed the last output tensor from the convolutiona base (of shape(4, 4 64) (shown in summary) into one or more
# Dense layers to perform classification. Dense layers take vectors as input (which are 1D), while the current output is a 3D tensor.
# First we need to flatten (or unrol) the 3D output to 1Dm then add one or more Dense layers on top. CIFAR has 10 classes, so we can use a final Dense Layer with 10 outputs

# dense layers expect a flat list of numbers, but the base outputs a 3D tensor of shape (4, 4, 64)
# Flatten unrolls it into one long vector of 1024 elements (4 * 4 * 64 = 1024)
model.add(layers.Flatten())

# a Dense layer (also called a fully connected layer) is where every input connects to every neuron. Each of the 64 neurons computes a weighted sum of all 1024 inputs, adds a bias
# then applies the ReLU activation function
model.add(layers.Dense(64, activation='relu'))

# the output layer. Has 10 neurons, one for each class
model.add(layers.Dense(10))

model.summary()



# now we can compile and train the model

# we define the model but it has random weights and doesn't know how to learn yet, so we need to compile it to set up the learning rules
model.compile(optimizer='adam',  
# optimizer = adam (adaptive moment estimation) is the algorithm that adjusts the weights, after each batch, it works out which direction to nudge 
# each weight to reduce the error using gradients, then nudges it.

# loss is the score the model is trying to minimize., a number measuring how wrong the predictions are 
# cross entropy punishes the model based on the probability it gave to the correct class, where confidently wrong answers get hit harder
# categorical means there are multiple classes, and sparse means the labels are plain integers, which matches train_labels that holds values from 0-9
# the param from_logits=True means the model outputs raw numbers (logits) before applying softmax, which is more numerically stable
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

# metrics is what gets reported during training: the percentage of images classified correctly
            metrics=['accuracy'])

# the actual training loop, the model learns from the train_images and train_labels
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))


# to evaluate the model

plt.plot(history.history["accuracy"], label = "accuracy")
plt.plot(history.history["val_accuracy"], label = "val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")

