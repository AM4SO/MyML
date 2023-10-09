import game
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
tf.test.is_gpu_available()
class agent:
    def __init__(self, plr, env=None, render=True, epsilon = 0.1):
        self.plrNum = plr
        if env == None:
            self.env = game.game(6,7)
        else:
            self.env = env
        self.nodesToWin = 4
        self.reward = 0
        self.render = render
        self.state = self.env.getState(self.plrNum)
        self.done = False
        self.predictor = None
        self.createPredictor()
        self.epsilon = epsilon
        self.previousReward = 0
        self.previousPrediction = np.zeros((1,7))
        self.previousAction = 0
    
    def createPredictor(self):
        self.predictor = Sequential([
            keras.layers.Conv2D(80,(self.nodesToWin,self.nodesToWin), activation="relu", input_shape=self.state.shape),
            keras.layers.Flatten(),
            keras.layers.Dense(2000, activation="relu"),
            keras.layers.Dense(400, activation="relu"),
            keras.layers.Dense(self.env.cols, activation="tanh")
        ])
        opt = tf.keras.optimizers.Adam(lr=0.00000001)
        self.predictor.compile(loss="mean_absolute_error",optimizer=opt)
    def chooseAction(self):
        if np.random.random() < self.epsilon:
            self.previousAction = np.random.randint(self.env.cols)
            self.previousPrediction = self.predictor.predict(np.expand_dims(self.state, axis=0))
            return self.previousAction
        else:
            self.previousPrediction = self.predictor.predict(np.expand_dims(self.state, axis=0))
            self.previousAction = np.argmax(self.previousPrediction)
            return self.previousAction
    def learn(self):
        #pred = self.previousPrediction
        correct = self.previousPrediction
        correct[0,self.previousAction] = self.previousReward
        self.predictor.fit(x=np.expand_dims(self.state, axis=0), y=np.expand_dims(correct,axis=0), epochs=1)
    def step(self):
        self.previousReward = self.reward
        self.learn()
        self.state, self.reward, self.done = self.env.step(self.chooseAction(),  self.plrNum) # self.env.step(np.random.randint(7), self.plrNum)
        if self.render:
            self.env.show()

        return self.reward

env = game.game(6,7)
test0 = agent(0, env=env,render=False)
test1 = agent(1, env=env,render=False)
winner = 1
for i in range(1000):
    while not env.done:
        while test0.step() == -1:
            continue
        print()
        #time.sleep(0.5)
        if not env.done:
            while test1.step() == -1:
                continue
            print()
            #time.sleep(0.5)
        else:
            winner = 0
    if winner == 0:
        test1.step()
    if winner == 1:# Step one last time through the player that didn't win as they haven't
        test0.step()#  yet claimed their negative reward, meaning they wont learn.
    env.reset()