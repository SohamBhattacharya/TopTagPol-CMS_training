import numpy
import tensorflow

from tensorflow.keras import datasets, layers, models
from tensorflow.keras import mixed_precision


# No. of images
nImg = 1000

nBinX = 50
nBinY = 50

# No. of layers/channels
nLayer = 1

nPixelTot = nBinX*nBinY

l_idx = []
l_val = []
l_label = []

# Create a sparse dataset with random entries
for iImg in range(0, nImg) :
    
    # Fill at most 60 pixels
    nFill = numpy.random.randint(low = 1, high = 61)
    
    for iFill in range(0, nFill) :
        
        # Index of the filled pixel
        # [image idx, row idx, col idx, layer]
        idx = [
            iImg,
            numpy.random.randint(low = 0, high = nBinY),
            numpy.random.randint(low = 0, high = nBinX),
            nLayer-1,
        ]
        
        if (idx in l_idx) :
            continue
        
        l_idx.append(idx)
        l_val.append(numpy.random.rand())
    
    l_label.append(numpy.random.randint(low = 0, high = 2))


img_shape = (nBinY, nBinX, nLayer)
dense_shape = (nImg, nBinY, nBinX, nLayer)

# Create the sparse rensor
input_img_sparseTensor = tensorflow.sparse.reorder(tensorflow.sparse.SparseTensor(
    indices = l_idx,
    values = l_val,
    dense_shape = dense_shape,
))


print("=====> Creating dataset...")

dataset_img = tensorflow.data.Dataset.from_tensor_slices(input_img_sparseTensor)
dataset_label = tensorflow.data.Dataset.from_tensor_slices(l_label)

batch_size = 100

dataset = tensorflow.data.Dataset.zip((dataset_img, dataset_label)).batch(batch_size)

print("dataset.element_spec:", dataset.element_spec)
print("=====> Created dataset...")


# Dummy CNN model
model = models.Sequential()

##model.add(layers.InputLayer(input_shape = img_shape, sparse = True, batch_size = batch_size))
model.add(layers.Conv2D(10, kernel_size = (10, 10), activation = "relu", input_shape = img_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(5, kernel_size = (5, 5), activation = "relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dense(2, activation = "relu"))

model.summary()

print("=====> Compiling model...")

model.compile(
    optimizer = "adam",
    loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ["accuracy"],
)

print("=====> Compiled model...")

print("=====> Starting fit...")

# Use the same data for train and test, just to check if it runs
history = model.fit(
    x = dataset,
    epochs = 5,
    #batch_size = batch_size,
    validation_data = dataset,
    shuffle = False,
)
