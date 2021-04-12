# Simple CNN to classify CIFAR images using Keras Sequential API 
# CIFAR-10 Dataset is used for this network
# TensorFlow Library imported along with a helper library Matplotlib.

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
# Helper Library
import matplotlib.pyplot as plt
# Callback Class defined
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.75):
      print("\nReached 75% accuracy on training data so cancelling training!")
      self.model.stop_training = True

# Load Test data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

print("\n**********************************************************")
print("***Running CIFAR Convolutional Neural Network program!***") # Print line signifying what NN and dataset is being compiled
print("**********************************************************\n")

# Using MatPlot Library, the dataset is verified initially
# First 25 images are plotted from training set with the class names found below
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Callback function called
callbacks = myCallback()

# Convolutional base is created using a stack of Conv2D and MaxPooling2D layers.
# Tensors are of shape (image_height, image_width, color_channels) with channels being RGB
# The format for the CIFAR shape is chosen
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # First Convolution layer generating 32 filters,filters are 3x3, image size are 32x32 and x3 is used for the colour channels
model.add(layers.MaxPooling2D((2, 2))) # 2x2 pool, so every 4 pixels, the largest will remain
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Another convolutional layer on top of another pooling layer is added.
model.add(layers.MaxPooling2D((2, 2)))  # Layers has been quatered 2 times.

# Architecture of the model is displayed below
model.summary()
# The width and height dimensions should tend to shrink as you go deeper in the network

# Final output tensor from base needs to be fed into one or more Dense layers to perform classification.
# Dense layers take inputs as 1D so the current 3D shape from the convolution needs to be flattened before adding dense layer.
# CIFAR has 10 output classes so final dense layer of 10 is used.
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10)) 

print('\n New summary of architecture is printed\n')
model.summary()

# Model is compiled below
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels), callbacks=[callbacks])


# Model is then evaluated with MatPlotLib
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\n Test Accuracy: ', test_acc)
