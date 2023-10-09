import game
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow
import keras
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten

class agent:
    def __init__(self, env, plr, epsilon = 1, learn=True):
        self.env = env
        self.plr = plr
        self.gamma = 0.7
        self.state = None
        self.reward = None
        self.lastState = None
        self.rewardForLastState = None
        self.invalidMove = False
        self.startingEpsilon = epsilon
        self.epsilon = 1
        self.modelOutput = None
        self.maxTrainingData = 1000
        self.trainingData = []
        self.learning = learn
        #self.initNetwork()
        self.qTable = np.zeros((3**(self.env.rows*self.env.cols), self.env.cols))
        self.lastAction = None
        self.learningRate = 0.1
        self.action = None

    def reset(self):
        self.state = None
        self.reward = None
        self.lastState = None
        self.rewardForLastState = None
        self.invalidMove = False
        #self.epsilon = self.startingEpsilon
        self.modelOutput = None
        self.lastAction = None
        self.action = None
        self.trainingData = []

    def step(self):
        if self.reward == None:
            self.reward = 0
            #print("THing")
            #print(self.encodeState(self.env.gameboard))
        self.lastState = self.state
        self.state = self.env.gameboard.copy()
        if self.learning:
            self.getGreedyAction()

        #if type(self.lastAction) != type(None) and self.learning:
        #    self.updateQTable()

        self.rewardForLastState = self.reward
        self.lastAction = self.action

        self.action = self.chooseAction()
        s, self.reward, done = self.env.step(self.action, self.plr)

        #print(f"State: {self.encodeState(self.state)}")
        self.trainingData.append([self.encodeState(self.state), self.action, self.reward])

        self.invalidMove = self.reward == -3001

    def train(self, data): ## follows:  [ [  [state, action, reward], [state1, action1, reward1]  ], 
    #                                     [  [state, action, reward], [state1, action1, reward1]  ] ]
        for ep in data:
            totalReward = 0
            #print(ep)
            for t in range(len(ep)-1, -1, -1):
                state, action, reward = ep[t][0], ep[t][1], ep[t][2]
                nextStateVal = 0
                if t != len(ep)-1:
                    nextStateVal = self.qTable[ep[t+1][0], ep[t+1][1]]

                qVal = self.qTable[state, action]
                qTarget = reward + totalReward
                self.qTable[state, action] = qVal + self.learningRate * (qTarget - qVal)
                #print(f"t: {t}, qTarget: {qTarget}, newQ: {self.qTable[state, action]}, nextStateVal: {nextStateVal}, state: {state}")
                totalReward += reward
                totalReward *= self.gamma

    def updateQTable(self):
        lastState = self.lastState
        lastReward = self.rew
        prnitardForLastState
        state = self.state
        reward = self.reward

        thisStateVal = np.amax(self.modelOutput)  # value of greedy action in this state
        greedyAction = np.argmax(self.modelOutput)

        qTarget = lastReward + self.gamma * thisStateVal
        
        encodedLastState = self.encodeState(lastState)
        qVal = self.qTable[encodedLastState,self.lastAction]
        if lastReward == 0 and np.random.random() < 0.01:
            print(f" qVals: {self.qTable[encodedLastState]},  state: {encodedLastState},   lastReward: {lastReward}")
        self.qTable[encodedLastState,self.lastAction] = qVal + self.learningRate * (qTarget - qVal)

    def initNetwork(self):
        self.stateActionValModel = Sequential([
            Conv2D(20,(4,4), activation="relu", input_shape=self.env.state.shape[1:]),
            Conv2D(80, (4,4), activation="relu"),
            Flatten(),
            Dense(200, activation="relu"),
            Dense(200, activation="relu"),
            Dense(self.env.cols)#, activation="softmax")
        ])
        opt = keras.optimizers.Adam(lr = 0.1)
        self.stateActionValModel.compile(optimizer=opt, loss = "mse", metrics=["mae"])

    def encodeState(self, state):
        power = state.shape[0] * state.shape[1] - 1
        encoded = 0
        i, j = 0, 0
        while power >= 0:
            encoded += (state[i,j]+1) * 3**power
            power -= 1
            j += 1
            if j == state.shape[1]:
                j = 0
                i += 1
        return encoded

    def getGreedyAction(self):
        self.modelOutput = self.qTable[self.encodeState(self.state)]
        return np.argmax(self.modelOutput)

    def chooseAction(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.cols)
        return np.argmax(self.modelOutput)

print("SFJSFH")
env = game.game(3,4, nodesToWin = 3)
plr0 = agent(env, 0,0.8)
plr1 = agent(env, 1, 0.8)
iters = 10000
print("dshdjgd")
for i in range(iters):
    I = 0
    done = False
    theRealDone = False
    timeDone = 0
    plr0.epsilon *= 0.9999
    plr1.epsilon *= 0.9999
    #if i % 1000 == 0:
    #    print(plr0.epsilon)
    if i % 1000 == 0:
        print(i)
    while not theRealDone:
        p = plr0
        if I % 2 != 0:
            p = plr1
        p.step()
        #if (i+1) % 1000 == 0:
            #env.show()
        #if i == iters-1:
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
    plr0.train([plr0.trainingData])
    plr1.train([plr1.trainingData])
    #plr0.stateActionValModel.fit(plr0.trainingData[0],plr0.trainingData[1])
    #plr1.stateActionValModel.fit(plr1.trainingData[0],plr1.trainingData[1])
    plr0.reset()
    plr1.reset()
    env.reset()
    if i == iters - 1:
        plr0.epsilon = 0
        x = 0
        while not env.done:
            if x % 2 == 0:
                plr0.step()
                env.show()
                print(plr0.qTable[plr0.encodeState(plr0.state)])
            else:
                inp = "pp"
                validMoves = env.getLegalActions()
                while not inp.isnumeric() or not int(inp)-1 in validMoves:
                    inp=input("Where to place:  ")
                env.placeAt(int(inp)-1, 1)
                env.show()
                time.sleep(1)
            x+=1
x = open("test.txt", "w+")
np.set_printoptions(threshold=np.inf)
x.write(np.array_str(plr0.qTable))
np.set_printoptions(threshold=1000)
x.close()
print("Done")