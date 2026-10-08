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
# plt.show()


# now we create a convolutional base:
# The convolutional base is the first part of a CNN: the stack of Conv2D and MaxPooling2D layers
# that extract features from the image, not a special keras object, just a name for that section of the model

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()



