import keras
import tensorflow

from classification_models.tfkeras import Classifiers
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import mixed_precision

import utils
import utils_tensorflow



d_network = {}


def build_and_get_CNN1(input_shape, nCategory) :
    
    model = models.Sequential()
    model.add(layers.Lambda(function = utils_tensorflow.sparse_to_dense, input_shape = input_shape))
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
    model.add(layers.Lambda(function = utils_tensorflow.sparse_to_dense, input_shape = input_shape))
    model.add(layers.Conv2D(50, kernel_size = (10, 10), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(30, kernel_size = (5, 5), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(10, kernel_size = (2, 2), activation = "relu"))
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate = 0.5))
    model.add(layers.Dense(200, activation = "relu"))
    model.add(layers.Dropout(rate = 0.5))
    model.add(layers.Dense(200, activation = "relu"))
    model.add(layers.Dropout(rate = 0.5))
    model.add(layers.Dense(200, activation = "relu"))
    model.add(layers.Dropout(rate = 0.5))
    model.add(layers.Dense(nCategory, activation = "relu"))
    model.add(layers.Dense(nCategory, activation = tensorflow.keras.activations.softmax))
    
    return model

d_network["CNN2"] = build_and_get_CNN2


def build_and_get_ResNet50(input_shape, nCategory) :
    
    resnet50, preproc = Classifiers.get("resnet50")
    
    network_input = layers.Input(shape = input_shape)
    layer_spd = layers.Lambda(function = utils_tensorflow.sparse_to_dense, input_shape = input_shape)(network_input)
    
    base_model = resnet50(
        input_shape = input_shape,
        input_tensor = layer_spd,
        include_top = False,
        classes = nCategory
    )
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    
    x = layers.Dropout(rate = 0.5)(x)
    x = layers.Dense(nCategory, activation = "relu")(x)
    
    network_output = layers.Dense(nCategory, activation = tensorflow.keras.activations.softmax)(x)
    
    model = models.Model(inputs = base_model.input, outputs = network_output)
    
    return model

d_network["ResNet50"] = build_and_get_ResNet50


def build_and_get_ResNet50NoDOL(input_shape, nCategory) :
    
    resnet50, preproc = Classifiers.get("resnet50")
    
    network_input = layers.Input(shape = input_shape)
    layer_spd = layers.Lambda(function = utils_tensorflow.sparse_to_dense, input_shape = input_shape)(network_input)
    
    base_model = resnet50(
        input_shape = input_shape,
        input_tensor = layer_spd,
        include_top = False,
        classes = nCategory
    )
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    
    x = layers.Dense(nCategory, activation = "relu")(x)
    
    network_output = layers.Dense(nCategory, activation = tensorflow.keras.activations.softmax)(x)
    
    model = models.Model(inputs = base_model.input, outputs = network_output)
    
    return model

d_network["ResNet50NoDOL"] = build_and_get_ResNet50NoDOL


def build_and_get_ResNeXt50(input_shape, nCategory) :
    
    resnext50, preproc = Classifiers.get("resnext50")
    
    network_input = layers.Input(shape = input_shape)#, sparse = True)
    layer_spd = layers.Lambda(function = utils_tensorflow.sparse_to_dense, input_shape = input_shape)(network_input)
    
    base_model = resnext50(
        input_shape = input_shape,
        input_tensor = layer_spd,
        include_top = False,
        classes = nCategory,
        #groups = 16,
    )
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    
    x = layers.Dropout(rate = 0.5)(x)
    x = layers.Dense(nCategory, activation = "relu")(x)
    
    network_output = layers.Dense(nCategory, activation = tensorflow.keras.activations.softmax)(x)
    
    model = models.Model(inputs = base_model.input, outputs = network_output)
    
    return model

d_network["ResNeXt50"] = build_and_get_ResNeXt50


def build_and_get_ResNeXt50NoDOL(input_shape, nCategory) :
    
    resnext50, preproc = Classifiers.get("resnext50")
    
    network_input = layers.Input(shape = input_shape)#, sparse = True)
    layer_spd = layers.Lambda(function = utils_tensorflow.sparse_to_dense, input_shape = input_shape)(network_input)
    
    base_model = resnext50(
        input_shape = input_shape,
        input_tensor = layer_spd,
        include_top = False,
        classes = nCategory
    )
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    
    x = layers.Dense(nCategory, activation = "relu")(x)
    
    network_output = layers.Dense(nCategory, activation = tensorflow.keras.activations.softmax)(x)
    
    model = models.Model(inputs = base_model.input, outputs = network_output)
    
    return model

d_network["ResNeXt50NoDOL"] = build_and_get_ResNeXt50NoDOL
