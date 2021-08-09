import tensorflow

from tensorflow.keras import datasets, layers, models
from tensorflow.keras import mixed_precision

import utils



d_network = {}


def build_and_get_CNN1(input_shape, nCategory) :
    
    model = models.Sequential()
    model.add(layers.Lambda(function = utils.sparse_to_dense, input_shape = input_shape))
    model.add(layers.Conv2D(50, kernel_size = (10, 10), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(30, kernel_size = (5, 5), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(10, kernel_size = (2, 2), activation = "relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(200, activation = "relu"))
    model.add(layers.Dense(200, activation = "relu"))
    model.add(layers.Dense(200, activation = "relu"))
    model.add(layers.Dense(nCategory, activation = "relu"))
    model.add(layers.Dense(nCategory, activation = tensorflow.keras.activations.softmax))
    
    return model

d_network["CNN1"] = build_and_get_CNN1


def build_and_get_CNN2(input_shape, nCategory) :
    
    model = models.Sequential()
    model.add(layers.Lambda(function = utils.sparse_to_dense, input_shape = input_shape))
    model.add(layers.Conv2D(50, kernel_size = (10, 10), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(30, kernel_size = (5, 5), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(10, kernel_size = (2, 2), activation = "relu"))
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate = 0.2))
    model.add(layers.Dense(200, activation = "relu"))
    model.add(layers.Dropout(rate = 0.2))
    model.add(layers.Dense(200, activation = "relu"))
    model.add(layers.Dropout(rate = 0.2))
    model.add(layers.Dense(200, activation = "relu"))
    model.add(layers.Dense(nCategory, activation = "relu"))
    model.add(layers.Dense(nCategory, activation = tensorflow.keras.activations.softmax))
    
    return model

d_network["CNN2"] = build_and_get_CNN2
