# -*- coding: utf-8 -*-
"""Basic_classification_FashionMNIST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19BaD2Yhi6ebJeq0JbcEOqrYIyA93LCuV

https://www.tensorflow.org/tutorials/keras/basic_classification
"""

# importing tensorflow and Keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# download fashion mnist dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class names not included with the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Exploring the data

print(train_images.shape)
print(len(train_labels))
print(set(train_labels))
print(test_images.shape)
print(len(test_labels))

# For Data pre-processing

print(train_images[0].shape)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

# Rescaling the pixel values

train_images = train_images / 255.0

test_images = test_images / 255.0

# Display the first 25 images from the training set and display the class name below each image. Verify that the data 
# is in the correct format and we're ready to build and train the network.

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# Setting the Network layers and creating the model
from keras import regularizers
model = keras.Sequential([
    
    # The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a 2d-array 
    # (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels
    keras.layers.Flatten(input_shape=(28, 28)),
    
    # The first Dense layer has 128 nodes (or neurons). The second (and last) layer is a 10-node softmax layer—this returns 
    # an array of 10 probability scores that sum to 1. Each node contains a score that indicates the probability that the 
    # current image belongs to one of the 10 classes.
    keras.layers.Dense(128, 
                       activation=tf.nn.relu,
                       kernel_initializer='he_normal',
                       bias_initializer='zeros'),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model

model.fit(train_images, train_labels, epochs=50)

# Evaluating accuracy

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Making predictions

predictions = model.predict(test_images)

predictions[0]

# Returns the indices of the maximum values along an axis
print(np.argmax(predictions[0]))
print(test_labels[0])

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

# !apt-get -qq install -y graphviz && pip install -q pydot
import pydot

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

from keras.utils import plot_model
plot_model(model, to_file='model.png')

