import numpy as np
import gym
import random
import time
import math
import json


class program:
    gamma = None
    graphical = None
    evnType = "Taxi-v3"
    env = None
    state = None
    reward = None
    done = False
    x = None
    action = None
    epsilon = None
    beforeState = None
    learningRate = None
    actions = 6
    testing = False
    epsilonMultiplier = None
    totalReward = None
    
    def initQTable(self):
        for i in range(500):
            self.qTable2[i] = {}
            for j in range(self.actions):
                self.qTable2[i][j] = (0, -1)

    def __init__(self, gamma, epsilon, learningRate, graphical):
        self.gamma = gamma
        self.graphical = graphical
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.epsilonMultiplier = 1
        self.qTable2 = {}
        self.initQTable()

    def createEnv(self):
        self.env = gym.make(self.evnType).env
        self.state = self.env.reset()
        if self.graphical:
            self.env.render()

    def run(self, timeScale, maxSteps=10000000):
        steps = 0
        self.totalReward = 0
        timeSleep = 0
        if timeScale != 0:
            timeSleep = 1/(1 * timeScale)
        while not self.done and steps < maxSteps:
            self.step()
            self.totalReward += self.reward
            time.sleep(timeSleep)
            steps += 1
        self.done = False
    
    def updateQTable(self):

        lastState = self.beforeState
        action = self.action
        thisState = self.state
        reward = self.reward
        prevStateQ = self.qTable2[lastState][action][0]

        thisStateQ = 0
        greedyAction = self.chooseAction(thisState, 0)
        thisStateQ = self.qTable2[thisState][greedyAction][0]

        lr, gamma = self.learningRate, self.gamma
        
        Gt = reward + gamma*thisStateQ
        qVal = prevStateQ +  lr*( Gt - prevStateQ )

        self.qTable2[lastState][action] = [qVal, thisState]

    
    def getOptimalAction(self, state=None):
        if not state:
            state = self.state
        
        arr = self.qTable2[state]
        x = -1000
        index = -1
        for i in range(len(arr)):
            if arr[i][0] > x:
                x = arr[i][0]
                index = i
        return index
########################################################
    def chooseAction(self, state = None, epsilon=None):
        if not epsilon:
            epsilon = self.epsilon
        
        if not state:
            state = self.state
        
        action = random.randint(0,5)
        if random.random() > epsilon:
            action = self.getOptimalAction(state)
            

        return action
#################################
    def step(self):
        env = self.env
        
        self.action = self.chooseAction()
        self.beforeState = self.state
        self.state, self.reward, self.done, self.x = env.step(self.action)
        if not self.testing:
            self.updateQTable()
        if self.graphical:
            env.render()

###########################
    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.action = None
        self.reward = None
        self.steps = 0
        self.cumulativeReward = 0
        self.pathTaken = [[],[]]
        self.estCumulativeReward = 0
#############################
    def test(self, iters, timeScale):
        a, b = self.epsilon, self.graphical
        self.epsilon = 0
        self.graphical = True
        self.testing = True
        for i in range(iters):
            print("iteration " + str(i))
            self.run(timeScale, 30)
            self.reset()
        self.epsilon = a
        self.testing = False
        self.graphical = b
############################
    def saveModel(self):
        x = json.dumps(self.qTable2)
        f = open("ai.txt", "w+")
        f.write(x)
        f.close()
    def loadModel(self):
        model = open("ai.txt", "r")
        qTable2Str = json.loads(model.read())
        for i in range(len(qTable2Str)):
            for j in range(self.actions):
                self.qTable2[i][j] = qTable2Str[str(i)][str(j)]
        model.close()

train = True

startingEpsilon = 0.6

main = program(1, startingEpsilon, 0.5, False)
main.createEnv()

trainIters = 2000

if train:
    print("learning....")
    for i in range(trainIters):#100000
        main.epsilon -= startingEpsilon/trainIters#0.00001
        if i % 100 == 0:
            #main.test(1, 40)
            print("iter " + str(i))
            print(main.epsilon)
        main.run(0, 20000000)
        main.reset()
    main.epsilon = 0.1
    main.learningRate = 0.1
    for i in range(1000):
        if i % 100 == 0:
            print(i/100)
        main.run(0, 20000000)
        main.reset()
else:
    main.loadModel()
print("\n\n\n\ntesting....")
time.sleep(1)
main.test(60, 10)

main.saveModel()