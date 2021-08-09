from __future__ import print_function

#import mxnet
import numpy
import os
import psutil
import sys
import tensorflow
import time

from tensorflow.keras import datasets, layers, models

def getMemoryMB(process = -1) :
    
    if (process < 0) :
        
        process = psutil.Process(os.getpid())
    
    mem = process.memory_info().rss / 1024.0**2
    
    return mem


#arr_shape = (int(1e6), 3, 50, 50)
#
#np_array = numpy.ones(arr_shape, dtype = numpy.float32)
##print(sys.getsizeof(numpy_array))
#
##print(np_array.itemsize)
#print("Memory:", getMemoryMB())
#np_array = None
#print("Memory:", getMemoryMB())
#
##sp_array = mxnet.ndarray.sparse.csr_matrix((100000, 10000))
##sp_array = mxnet.ndarray.sparse.zeros("row_sparse", shape = (100000, 10000))
#
##sp_array[10, 100] = numpy.random.rand()
#
##time.sleep(100)
#
##print(sp_array.shape)
#
##print("Memory:", getMemoryMB())
#
#
#sp_array = tensorflow.sparse.SparseTensor(
#    indices = [[0, 0, 0, 0]],
#    values = [0.8],
#    dense_shape = arr_shape,
#)
#
#print(sp_array)
#
#print("Memory:", getMemoryMB())


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print(type(train_images))

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
print("")

train_images = tensorflow.sparse.from_dense(train_images)
#train_labels = tensorflow.sparse.from_dense(train_labels)
test_images = tensorflow.sparse.from_dense(test_images)
#test_labels = tensorflow.sparse.from_dense(test_labels)

print(train_images)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

def sparse_to_dense(tensor) :
    
    print(tensor)
    print(type(tensor))
    print(tensor.get_shape())
    
    #if (isinstance(tensor, tensorflow.sparse.SparseTensor) and tensor.get_shape()[0] is not None and tensor.get_shape()[1:] == (32, 32, 3)) :
    if (isinstance(tensor, tensorflow.sparse.SparseTensor)) :
        
        print("Converting to dense")
        dn_tensor = tensorflow.sparse.to_dense(tensor)#.numpy()
        print(dn_tensor)
        print(type(dn_tensor))
        
        return dn_tensor
    
    return tensor

batch_size = 100

model = models.Sequential()
#inputLayer = layers.Input(shape = (32, 32, 3), sparse = True)#, batch_size = train_images.shape[0]))
#model.add(layers.Lambda(tensorflow.sparse.to_dense)(inputLayer))
#model.add(layers.Lambda(lambda x: tensorflow.sparse.to_dense(x)))
#model.add(layers.Lambda(function = sparse_to_dense, input_shape = (32, 32, 3)))
#inp = layers.Input(type_spec = tensorflow.TensorSpec(shape = (batch_size, 32, 32, 3)))#, dtype = tensorflow.dtypes.int64))
#model.add(layers.Conv2D(32, (3, 3), activation = "relu")(inp))#, input_shape = (32, 32, 3)))
#model.add(layers.InputLayer(type_spec = tensorflow.TensorSpec(shape = (batch_size, 32, 32, 3))))

model.add(layers.Lambda(function = sparse_to_dense, input_shape = (32, 32, 3)))

model.add(layers.Conv2D(32, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(10))

model.summary()

model.compile(
    optimizer = "adam",
    loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ["accuracy"],
    #run_eagerly = True,
)

history = model.fit(
    train_images,
    train_labels,
    epochs = 10,
    batch_size = batch_size,
    validation_data = (test_images, test_labels),
)



