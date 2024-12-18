import numpy as np

class game:
  """
  A class for a customisable connect-4 game.

  
  Methods:
    getState\n
    show\n
    reset\n
    step

    
  Attributes:
    done, rows, cols, nodesToWin, gameboard, state, winSlice

  Play the game by calling game.step(), whilst switching between the players.
  Repeat this until game.done == True. At this point, a player must have won
  or a player must have tried to play after the board was already filled. Call
  game.step once more for the other player, so that both players have been 
  told that the game has ended. 
  """
  def __init__(self, rows:int, cols:int, nodesToWin = 4):
    """
    Constructor(rows, columns, nodesToWin=4)
    """
    self.done = False
    self.rows = rows
    self.cols = cols
    self.nodesToWin = nodesToWin
    self.gameboard = -np.ones((rows, cols),dtype=np.int8)
    self.winSlice = [[0]*nodesToWin,[1]*nodesToWin]
    self.state = np.zeros((2, 4*rows,cols,1),dtype = np.int8)#rowSet0 = empty, rowSet1=localPlr, rowSet2=OtherPlayer, rowSet4=AnythingThere. 1 if plr0 or plr1
    self.winner = -1

    self.winReward = 100
    self.loseReward = -100
    self.incorrectInputReward = -1
    self.drawReward = 0
    self.normalReward = 0
  
  def peekActions(self, actions:np.ndarray, plr:int) -> np.ndarray:
    """Returns the states that follow each given action"""
    ret = np.tile(self.gameboard, (len(actions), 1, 1) )
    legalMoves = self.getLegalMoves()
    for i, action in enumerate(actions):
      assert action in legalMoves, f"Cannot peek illegal action {action}"

      ret[i, self.findBottomFree(action), action] = plr
    return ret

  def getLegalMoves(self):
    ret = np.where(self.gameboard[0,:] == -1)[0]
    if len(ret) == 0:
      return np.array([-1], dtype = np.intp)
    return ret
    
  def checkPos(self, pos):
    """
    checkPos((X, Y))
    Returns true if the given position is empty
    """
    if self.gameboard[pos[0],pos[1]] == -1:
      return True
    return False

  def getCol(self, pos):
    return self.gameboard[:, pos].copy()
  
  def getDiags(self, loc):
    row, col = loc
    c = row-col#C in formula of diag y=mx+c
    diag = []
    for col in range(self.cols):
      row = col + c
      if row >= 0 and row < self.rows:
        diag.append(self.gameboard[row,col])
    row, col = loc
    diag1 = []
    rRow = self.rows - row - 1
    c = rRow - col
    for col in range(self.cols):
      rRow = col + c
      row = self.rows - rRow - 1
      if row >= 0 and row < self.rows:
        diag1.append(self.gameboard[row,col])
    return [diag, diag1]

  def checkWin(self, plr:int, placed:tuple):
    row = self.gameboard[placed[0]]
    col = self.getCol(placed[1])
    win = self.winSlice[plr]
    if self.XinY(win, row) or self.XinY(win, col):
      self.done = True
      return True
    diags = self.getDiags((placed[0], placed[1]))
    if self.XinY(win, diags[0]) or self.XinY(win, diags[1]):
      self.done = True
      return True
    return False
  
  def step(self, action:int, plr:int) -> tuple["state", int, bool]:
    if self.done:#other player won on their last turn
      return (self.getState(plr), self.loseReward, True)
    x = self.placeAt(action, plr)
    if x == -10:
        return (self.getState(plr), self.incorrectInputReward, False)
    if x == -1:
        return (self.getState(plr), self.normalReward, False)
    if x == plr:
        self.winner = plr
        return (self.getState(plr), self.winReward, True)
    if x == -100:
        return (self.getState(plr), self.drawReward, True)
    
  def findBottomFree(self, column:int) -> int:
    return np.where(self.gameboard[:, column] == -1)[0][-1]

  def placeAt(self, pos:int, plr:int):
    col = self.getCol(pos)

    if not -1 in self.gameboard: # game board is already full
        self.done = True
        return -100
    
    if -1 in col and pos >= 0 and pos < self.cols:
      coords = [self.findBottomFree(pos), pos]

      self.gameboard[coords[0],coords[1]] = plr

      if(self.checkWin(plr, coords)):
        return plr
    else:
      print("NO")
      return -10 # Incorrect input
    return -1 ## Normal return
  def getState(self, plr=None):
    return self.gameboard.copy()

    if plr == 0:
       return self.state[0]
    return self.state[1]  
  def XinY(self, x, y):
    i = 0
    xLen = len(x)
    for j in range(len(y)):
      if i != xLen:
        if y[j] == x[i]:
          i+=1
        else:
          i = 0
      if i == xLen:
        return True
    return False

  def show(self):
    """
    Prints the gameboard to the screen
    """
    board = []
    thing = []
    for i in range(self.rows):
        board.append([])
        for j in range(self.cols):
            if i == 0:
                thing.append("|---|")
            if self.gameboard[i, j] == -1:
                board[i].append("|   |")
            elif self.gameboard[i,j] == 0:
                board[i].append("| o |")
            else:
                board[i].append("| x |")
    for row in board:
        string = ""
        for v in thing:
            string+=v
        print(string)
        string = ""
        for v in row:
            string+=v
        print(string)
  def reset(self):
      self.__init__(self.rows, self.cols, self.nodesToWin)

