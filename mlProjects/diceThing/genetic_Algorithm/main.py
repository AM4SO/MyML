
"""
This one failed as because, for it to progress, at least 1 agent needs to be able to get at least one episode correct, and since it will always take the same action for one specific state that it is in, it is very 
highly unlikely that it will have a combination of state - action values which are perfect and able to complete the episode. This means that it will never really manage to get one episode correct which means that it 
will never be able to start learning.
Probably shouldn't have spent time writing the reproduce algorithm before testing the idea; I never even got to use the reproduce function so it was a complete waste.
"""

import numpy as np
import gym
import math
import time
import json
import random

class agent:
    def __init__(self, states, actions, maxIters):
        self.states = states
        self.actions = actions
        self.actionTable = np.round(np.random.random(states) * (actions - 1))
        self.sliceLen = states / 2
        self.maxIters = maxIters
        self.initEnv()
    
    def initEnv(self):
        self.env = gym.make("Taxi-v3").env
        self.state = self.env.reset()
        self.stepCounter = 0
        self.rewardCounter = 0
        self.done = False
        self.success = False
    
    def step(self):
        if self.stepCounter <= self.maxIters and not self.done:
            self.state, self.reward, self.done, self.x = self.env.step(self.actionTable[self.state])
            self.stepCounter += 1
            self.rewardCounter += self.reward
            if self.done:
                self.success = True
        else:
            self.done = True

    def run(self):
        while not self.done:
            self.step()
        return (self.success, self.rewardCounter)
        
    def reproduce(self, otherActionTable, sliceMode="bioSlice"):
        if sliceMode == "bioSclice":
            sliceStart = random.randint(0,self.actions)
            sliceEnd = sliceStart + self.sliceLen
            secondSliceStart, secondSliceEnd = None, None
            if sliceEnd >= self.states:
                leftOver = (sliceEnd - self.states) + 1
                secondSliceStart = 0
                secondSliceEnd = leftOver
            
            newArr = np.ndarray(self.states)
            if secondSliceStart:
                newArr[secondSliceStart: secondSliceEnd] = otherActionTable[secondSliceStart:secondSliceEnd]
            
            newArr[sliceStart: sliceEnd] = otherActionTable[sliceStart: sliceEnd]
            self.actionTable = newArr
        elif sliceMode == "randSlice":
            newArr = np.ndarray(self.states)
            for i in range(self.states):
                if random.random() < 0.5:
                    newArr.append(self.actionTable[i])
                else:
                    newArr.append(otherActionTable[i])
            self.actionTable = newArr
        return self
    
class population:
    def __init__(self, size, states, actions):
        self.maxPopSize = size
        self.states = states
        self.actions = actions
        self.initPopulation()
    
    def initPopulation(self):
        self.population = []
        for i in range(self.maxPopSize):
            self.population.append(agent(self.states, self.actions))
        
for i in range(10000):
    agent0 = agent(500, 6, 60)
    x = agent0.run()
    print(x)
