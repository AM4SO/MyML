import game
import numpy as np
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow
import keras
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.callbacks import History

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
        self.lastLoss = 10000

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

        self.invalidMove = self.reward == -90
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
            #i = 0
            modelInputs = None
            print("Getting states to pass to model")
            for ep in data:
                for t in range(len(ep)-1, -1, -1): ## gather all of the states of the episode into an array to pass
                    state = ep[t][0]               ## to the model, which will predict action values. We correct these
    #                                              ## action values to our estimate of the correct values.
                    if type(modelInputs) == type(None):
                        modelInputs = np.expand_dims(state, axis=0)
                    else:
                        modelInputs = np.concatenate((modelInputs,np.expand_dims(state, axis=0)), axis=0)

            print("Predicting values")
            modelOutputs = self.stateActionValModel.predict(modelInputs)
            j = 0
            print("Creating training data")
            for ep in data:
                totalReward = 0
                for t in range(len(ep)-1, -1, -1):
                    state, action, reward = ep[t][0], ep[t][1], ep[t][2]
                    nextStateVal = 0
                    if t+1 < len(ep):
                        nextStateVal = np.mean(modelOutputs[j-1]) # val of next state is mean of action vals in next state
                    modelOutputs[j,action] = reward + self.gamma*nextStateVal#totalReward
                    
                    if type(trainX) == type(None):
                        trainX = np.expand_dims(state, axis=0)
                        trainY = np.expand_dims(modelOutputs[j], axis=0)
                    else:
                        trainX = np.concatenate(   ( trainX, np.expand_dims(state, axis=0) )  , axis=0)
                        trainY = np.concatenate(   (trainY, np.expand_dims(modelOutputs[j],axis=0)), axis = 0)

                    totalReward += reward * self.gamma
                    j+=1


                if i % 10 == 0:
                    print(i)
                i+=1
            acceptableResult = False
            count = 0
            while not acceptableResult:
                h = self.stateActionValModel.fit(trainX, trainY, epochs = 30,batch_size=round(trainX.shape[0]/2))
                acceptableResult = True#h.history["loss"][-1] < 100 or count >= 0
                count +=1
                self.lastLoss = h.history["loss"][-1]

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
            Flatten(),
            Dense(50, activation = "relu", input_shape = inShape),
            Dense(50, activation="relu"),
            Dense(50, activation="relu"),
            Dense(50, activation="relu"),
            Dense(50, activation="relu"),
            Dense(50, activation="relu"),
            Dense(self.env.cols)#, activation="softmax")
        ])
        opt = keras.optimizers.Adam(lr = 0.03)
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
iters = 1
thing = 20 # 60
for q in range(thing):
    plr0TrainData = []
    plr1TrainData = []
    print(q)
    for i in range(iters):
        #if q > 24:
        #    plr0.epsilon = 0.3
        #if i % 10 == 0:
        #    print(i)
        I = 0
        done = False
        theRealDone = False
        timeDone = 0
        while not theRealDone:
            p = plr0
            if I % 2 != 0:
                p = plr1
            p.step()
            #if (i+1) % 10 == 0:
            #    env.show()
            #if i == iters-1:
            #    env.show()
            #    time.sleep(1)
            #input()
            if p.invalidMove and not len(env.getLegalActions()) == 0:    #  invalid move
                #print("Invalid move")
                continue         # skip out on adding 1 to I
            if done:
                timeDone+=1
            done = env.done
            theRealDone = done and timeDone == 1
            I+=1
        plr0TrainData.append(plr0.trainingData.copy())
        plr1TrainData.append(plr1.trainingData.copy())
        #plr0.stateActionValModel.fit(plr0.trainingData[0],plr0.trainingData[1])
        #plr1.stateActionValModel.fit(plr1.trainingData[0],plr1.trainingData[1])
        plr0.reset()
        plr1.reset()
        env.reset()
        
    plr0.train(plr0TrainData)
    plr1.train(plr1TrainData)
plr0.epsilon = 0
x = 0
while not env.done:
    if x % 2 == 0:
        plr0.step()
        env.show()
        print(plr0.modelOutput)
    else:
        inp = "pp"
        validMoves = env.getLegalActions()
        while not inp.isnumeric() or not int(inp)-1 in validMoves:
            inp=input("Where to place:  ")
        env.placeAt(int(inp)-1, 1)
        env.show()
        time.sleep(1)
    x+=1
plr0.reset()
plr1.reset()
env.reset()

print("Done")