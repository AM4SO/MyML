import struct
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import pydot

tf.test.is_gpu_available()

def readIDX(path, dims):
    with open(path, "rb") as f:
        magic, dsSize = struct.unpack( ">II", f.read(8))
        if(dims == 3):
            rows, cols = struct.unpack(">II", f.read(8))
        if dims == 1:
            shape = (dsSize)
        else:
            shape = (dsSize, rows, cols)
        data = np.fromfile(f, dtype = np.dtype(np.uint8)).reshape(shape)
    return data

def fixLabels(labels):
    retVal = np.ndarray((labels.shape[0], 9))
    for i in range(labels.shape[0]):
        exampleResult = np.ndarray(1)
        for j in range(9):
            if j == labels[i]:
                retVal[i,j] = 1
            else:
                retVal[i,j] = 0
    return retVal

dataSetDir = "Dataset/"
trainImg0 = readIDX(dataSetDir + "train-images", 3)/255
print(0)
trainLabels = fixLabels(readIDX(dataSetDir + "train-labels", 1))
print(1)
testImg = readIDX(dataSetDir + "t10k-images", 3)/255
testLabels = readIDX(dataSetDir + "t10k-labels", 1)

model = keras.Sequential([
    layers.Conv2D(3, (5,5),activation="relu", input_shape=(28,28,1)),
    layers.Conv2D(12, (5,5),activation="relu"),
    #layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(6, (5,5),activation="relu"),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(1024, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(120, activation="relu"),
    layers.Dense(60, activation="relu"),
    layers.Dense(9, activation="softmax")
])
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.000001,#
    decay_steps=500,#4
    decay_rate=0.6)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='categorical_crossentropy', optimizer=opt)
trainImg = np.expand_dims(trainImg0, axis=3)
testImg = np.expand_dims(testImg, axis=3)
m = fixLabels(testLabels)
callbacks = keras.callbacks.TensorBoard()
#with tf.device("/gpu:0"):
model.fit(x=trainImg, y=trainLabels, batch_size=100, epochs=5, validation_data = (testImg,m), callbacks = [callbacks])

tests = 10000
l = 0
predictions = model.predict(testImg[0:tests])
wrongList = []
right = None
for i in range(tests):
    if( np.argmax(predictions[i]) == np.argmax(m[i])):
        l+=1
        right = [np.argmax(predictions[i]), testImg[i,:,:,0]]
    else:
        wrongList.append([np.argmax(predictions[i]), testImg[i,:,:,0]])
print(l/tests)
for v in wrongList:
    print(v[0])
    plt.imshow(v[1],cmap="gray")
    plt.show()
#keras.utils.plot_model(model)