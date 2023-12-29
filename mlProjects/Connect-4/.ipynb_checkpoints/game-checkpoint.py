import numpy as np

class game:
  def __init__(self, rows, cols, nodesToWin = 4):
    """
    Constructor(rows, columns, nodesToWin=4)
    """
    self.done = False
    self.rows = rows
    self.cols = cols
    self.nodesToWin = nodesToWin
    self.gameboard = -np.ones((rows, cols),dtype=np.int8)
    self.winSlice = [[],[]]
    self.state = np.zeros((2, 4*rows,cols,1),dtype = np.int8)#rowSet0 = empty, rowSet1=localPlr, rowSet2=OtherPlayer, rowSet4=AnythingThere. 1 if plr0 or plr1
    for i in range(nodesToWin):
      self.winSlice[0].append(0)
      self.winSlice[1].append(1)
  
    
  def checkPos(self, pos):
      """
      checkPos((X, Y))
      Returns true if the given position is empty
      """
    if self.gameboard[pos[0],pos[1]] == -1:
      return True
    return False

  def getCol(self, pos):
    col = []
    for row in range(self.rows):
      col.append(self.gameboard[row, pos])
    return col
  def getCoords(self, pos):
    col = self.getCol(pos)
    row = -1
    return [col, row]
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

  def checkWin(self, plr, placed):
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
  
  def step(self, action, plr):
    if self.done:#other player won on their last turn
        return (self.getState(plr), -1, True)
    x = self.placeAt(action, plr)
    if x == -10:
        return (self.getState(plr), -0.3, False)
    if x == -1:
        return (self.getState(plr), 0, False)
    if x == plr:
        return (self.getState(plr), 1, True)
    if x == -100:
        return (self.getState(plr), 0, True)
  def placeAt(self, pos, plr):
    col = self.getCol(pos)
    if not -1 in self.gameboard:
        self.done = True
        return -100
    if -1 in col and pos >= 0 and pos < self.cols:
      coords = [-1, pos]
      for i in range(self.rows):
        row = self.rows - 1 - i
        if col[row] == -1:
          coords[0] = row
          break
      self.gameboard[coords[0],coords[1]] = plr
      self.state[0,coords[0]+self.rows*(plr+1), coords[1],0] = 1
      self.state[1,coords[0]+(self.rows*3-self.rows*(plr+1)), coords[1],0] = 1
      
      self.state[0,coords[0], coords[1],0] = 0
      self.state[1,coords[0], coords[1],0] = 0
      self.state[0,coords[0]+self.rows*3, coords[1], 0] = 1
      self.state[1,coords[0]+self.rows*3, coords[1], 0] = 1
      if(self.checkWin(plr, coords)):
        return plr
    else:
      print("NO")
      return -10 # Incorrect input
    return -1 ## Normal return
  def getState(self, plr):
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

