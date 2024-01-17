# FLOWER-RECOGNITION-USING-CNN-WITH-PARALLELIZATION
FLOWER RECOGNITION USING CNN WITH PARALLELIZATION TO CALCULATE SPEED-UP &amp; EFFICIENCY

Creating a flower recognition system entails training a machine to identify various types of flowers. When the machine struggles to recognize flowers on its own, we step in to assist by training  a dataset. This process   involves adapting Convolutional Neural Networks (CNN), a widely  used approach  in image recognition tasks. CNNs made  up of several layers that work together to extract and learn features from images. Hence they are ideal for tasks like object recognition and hence flower recognition. 

Training a CNN can be  computationally expensive, especially for  large datasets. Parallelization  techniques can significantly  speed up the process by distributing  the workload across multiple CPUs, GPUs, or even cloud computing resources.

# PARALLELIZATION

## **TARGET** 
Training and running CNNs can be computationally expensive, especially for large datasets or complex models. 
Parallelization aims to distribute the workload across multiple processing units (multi-core CPU, GPU, or a cluster of machines) to speed up the process.



# Installations

* pip install -U scikit-learn
* pip install opencv-python
* pip install tensorflow tensorflow-datasets matplatlib


# Modules

* SCIKITLEARN
* TensorFlow
* OPENCV

# Libraries

* import tensorflow as tf
* import tensorflow_datasets as tfds
* import time
* import numpy as np
* from sklearn.metrics import confusion_matrix, classification_report
* import seaborn as sns
* import matplotlib.pyplot as plt
* from tensorflow.keras.models import Sequential
* from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense


# Convolutional Neural Network

For CNNs to function, an input image must travel through several convolutional layers. A set of filters is applied in each convolutional layer to the input picture to identify various features like edges, corners, and textures. A non-linear activation function, such as the Rectified Linear Unit (Relu), is then applied to the output of each convolutional layer to help bring nonlinearity into the network and enhance its capacity to learn complex features.

# Performace
It begins at around 60% and steadily increases to close to 100%, which suggests that the model fits well to the training set.



