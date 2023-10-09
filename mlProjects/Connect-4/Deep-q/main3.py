import game
import numpy as np
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow
import keras
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten

class agent:
    def __init__(self, env, plr, epsilon = 1, learn=True):
        self.env = env
        self.plr = plr
        self.gamma = 0.84
        self.state = None
        self.reward = None
        self.lastState = None
        self.rewardForLastState = None
        self.invalidMove = False
        self.startingEpsilon = epsilon
        self.epsilon = epsilon
        self.modelOutput = None
        self.maxTrainingData = 1000
        self.trainingData = []
        self.learning = learn
        self.initNetwork()
        self.action = None

    def reset(self):
        self.state = None
        self.reward = None
        self.lastState = None
        self.rewardForLastState = None
        self.invalidMove = False
        self.epsilon = self.startingEpsilon
        self.modelOutput = None
        self.action = None
        self.trainingData = []

    def step(self):
        if self.reward == None:
            self.reward = 0

        self.lastState = self.state
        self.state = self.createStateFromGameboard(self.env.gameboard.copy())
        if self.learning and self.epsilon < 1:
            self.getGreedyAction()
        
        self.rewardForLastState = self.reward
        self.lastAction = self.action

        self.action = self.chooseAction()
        s, self.reward, done = self.env.step(self.action, self.plr)
        
        self.trainingData.append([self.state, self.action, self.reward])

        #if type(self.lastState) != type(None) and self.learning:
        #    self.updateQTable()

        self.invalidMove = self.reward == -10
    def createStateFromGameboard(self, gameboard):
        retVal = np.zeros((gameboard.shape[0]*2+2, gameboard.shape[1]+2))
        for i in range(gameboard.shape[0]):
            for j in range(gameboard.shape[1]):
                #if gameboard[i,j] == -1:
                #    gameboard[i,j] = 0
                #elif gameboard[i,j] == 0:
                #    gameboard[i,j] = 1
                #elif gameboard[i,j] == 1:
                #    gameboard[i,j] = -1
                if i == 0 or i == retVal.shape[0]-1 or j == 0 or j == retVal.shape[1]:
                    gameboard[i,j] = -1
                    continue
                if gameboard[i,j] == 1:
                    retVal[gameboard.shape[0] + i, j] = 1
                elif gameboard[i,j] == 0:
                    retVal[i,j] = 1
        #print(retVal)
        return np.expand_dims(retVal, axis=2)
    def train(self, data):
        for i in range(1):
            trainX = None
            trainY = None
            for ep in data:
                

    def updateQTable(self):
        lastState = self.lastState
        lastReward = self.rewardForLastState
        state = self.state
        reward = self.reward

        thisStateVal = np.amax(self.modelOutput)  # value of greedy action in this state
        greedyAction = np.argmax(self.modelOutput)

        qTarget = lastReward + self.gamma * thisStateVal
        if np.random.random() < 0:
            print(f"plr_{self.plr} qTarget: {qTarget}")
        
        desiredOutput = self.modelOutput.copy()
        desiredOutput[greedyAction] = thisStateVal
        desiredOutput = np.expand_dims(desiredOutput, axis=0)

        modelInput = np.expand_dims(state,axis=0)
        if len(self.trainingData) == 0:
            self.trainingData.append(modelInput)
            self.trainingData.append(desiredOutput)
        else:
            self.trainingData[0] = np.concatenate((self.trainingData[0], modelInput),axis=0)
            self.trainingData[1] = np.concatenate((self.trainingData[1], desiredOutput),axis=0)
            if self.trainingData[0].shape[0] > self.maxTrainingData:
                #print("MAX SIZE REACHED")
                self.trainingData[0] = self.trainingData[0][1:]
                self.trainingData[1] = self.trainingData[1][1:]
        #self.stateActionValModel.fit(modelInput , desiredOutput)

    def initNetwork(self):
        inShape = np.expand_dims(self.env.gameboard, axis=2).shape
        inShape = (inShape[0]*2+2, inShape[1]+2, inShape[2])
        self.stateActionValModel = Sequential([
            Conv2D(60,(4,4), activation="relu", input_shape=inShape),
            #Conv2D(10, (3,3), activation="relu"),
            Flatten(),
            Dense(800, activation="relu"),
            Dense(800, activation="relu"),
            Dense(self.env.cols)#, activation="softmax")
        ])
        opt = keras.optimizers.Adam(lr = 0.001)
        self.stateActionValModel.compile(optimizer=opt, loss = "mse", metrics=["mae"])

    def getGreedyAction(self):
        modelOutput = self.stateActionValModel.predict(np.expand_dims(self.state, axis = 0))
        self.modelOutput = modelOutput[0]
        return np.argmax(modelOutput[0])

    def chooseAction(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.cols)
        return np.argmax(self.modelOutput)

env = game.game(6,7)
plr0 = agent(env, 0, 0.3)
plr1 = agent(env, 1, 0.3)
plr0.epsilon = 0
x = 0

plr0.reset()
plr1.reset()
env.reset()

iters = 1000
trainingData = None
for i in range(iters):
    I = 0
    done = False
    timeDone = 0
    theRealDone = False
    while not theRealDone:
        p = plr0
        if I % 2 != 0:
            p = plr1
        p.step()
        if p.invalidMove and not len(env.getLegalActions()) == 0:    #  invalid move
            continue
        if done:
            timeDone += 1
        done = env.done
        theRealDone = timeDone == 1
        I+=1

print("Done")