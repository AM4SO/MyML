import gym
import math

class DiceGame:
    metadata = {"render_modes" : ["human", "none"]}
    requestedAction = 0

    def __init__(self, boardSize=9, actionsPerMove=2):
               # 1,2,3,4,5,6,7,8,9
        self.board = [1 for i in range(boardSize)]
        self.actionsPerMove = actionsPerMove
        self.boardSize = boardSize

    def reset(self):
        self.board = [1 for i in range(boardSize)]

    def encodeAction(self, actionArr): ## [1,5] would be pushing down the 1 and the 5 dominoes
        actionsPerMove = self.actionsPerMove
        boardSize = self.boardSize
        actionArr.sort(reverse = True)
        if type(actionArr) != type([]) or len(actionArr) != actionsPerMove:
            raise ValueError(f"Input must be an array of length {actionsPerMove}")
            return
        for v in actionArr:
            assert 0 <= v and v <= boardSize
        ret = 0
        for i in range(actionsPerMove):
            ret += (boardSize+1) ** i * (actionArr[i])
        return ret

    def decodeAction(self, act:int):
        actionsPerMove = self.actionsPerMove
        boardSize = self.boardSize
        powerOfBS = actionsPerMove-1#math.floor(math.log(act, boardSize))
        retVal = []
        while powerOfBS >= 0:
            x = math.floor(act / ((boardSize+1)**powerOfBS))
            retVal.append(x)
            act -= x * ((boardSize+1)**powerOfBS)
            powerOfBS -= 1
        retVal.sort(reverse = True)
        return retVal
    
    def encodeState(self, board, targetSum):
        board = board.copy()
        board.append(targetSum)
        ret = 0

        for i in range(len(board)):
            ret += board[i] * 2**i
        return ret
    def decodeState(self, state):
        power = len(self.board)+1
        ret = []
        while power >= 0:
            x = math.floor(state / 2**power)
            ret.append(x)
            state -= x*2**power
            power-=1
        return ret

    def getValidActions(self, state=None):
        state = self.state if state == None
        

    def step(self, action):
        board = self.board
        action = self.decodeAction(action)
        test = [action[i] <= boardSize and action[i] >= 0 for i in range(len(action))]
        test = [test[i] and board[action[i]-1] == 1 for i in range(len(action))]
        total = 0
        total += v for v in action
        if False in test or total != requestedAction:
            raise ValueError("Invalid action")
        

    def render(self, mode = "human"):
        
