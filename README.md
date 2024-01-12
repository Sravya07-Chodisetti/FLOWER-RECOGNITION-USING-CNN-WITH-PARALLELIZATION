# FLOWER-RECOGNITION-USING-CNN-WITH-PARALLELIZATION
FLOWER RECOGNITION USING CNN WITH PARALLELIZATION TO CALCULATE SPEED-UP &amp; EFFICIENCY

Creating a flower recognition system entails training a machine to identify various types of flowers. When the machine struggles to recognize flowers on its own, we step in to assist by training  a dataset. This process   involves adapting Convolutional Neural Networks (CNN), a widely  used approach  in image recognition tasks. CNNs made  up of several layers that work together to extract and learn features from images. Hence they are ideal for tasks like object recognition and hence flower recognition. 

Training a CNN can be  computationally expensive, especially for  large datasets. Parallelization  techniques can significantly  speed up the process by distributing  the workload across multiple CPUs, GPUs, or even cloud computing resources.


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


