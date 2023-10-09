import numpy as np
import tensorflow
import keras
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten
from matplotlib import pyplot as plt

def createNN(layers, nodesPerLayer):
    x = [
        Dense(nodesPerLayer, activation="tanh") for i in range(layers-1)
    ]
    x.append(Dense(1, activation="tanh"))
    model = Sequential(x)
    model.compile(optimizer="Adam", loss = "mse", metrics=["mae"])
    return model

model = createNN(10, 40)

trainInputs = np.expand_dims(np.arange(-63,63,0.5),1)
trainOutputs = np.sin(trainInputs*0.1)#10*(np.random.random((400,1))-0.5)

plt.plot(trainInputs, trainOutputs)

inputs = np.expand_dims(np.arange(-63,63,0.5),1)
#outputs = model.predict(inputs)
#plt.plot(inputs,outputs)
for i in range(1):
    model.fit(trainInputs,trainOutputs, epochs = 500)
    outputs = model.predict(inputs)
    plt.plot(inputs, outputs)

plt.show()