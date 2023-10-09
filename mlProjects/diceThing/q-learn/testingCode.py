import math
def encodeAction(actionArr): ## [1,5] would be pushing down the 1 and the 5 dominoes
        actionsPerMove = 2
        boardSize = 9
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

def decodeAction(act:int):
        actionsPerMove = 2
        boardSize = 9
        powerOfBS = actionsPerMove-1#math.floor(math.log(act, boardSize))
        retVal = []
        while powerOfBS >= 0:
            x = math.floor(act / ((boardSize+1)**powerOfBS))
            retVal.append(x)
            act -= x * ((boardSize+1)**powerOfBS)
            powerOfBS -= 1
        retVal.sort(reverse = True)
        return retVal

#print(decodeAction(encodeAction([1,2])))
#print(decodeAction(encodeAction([2,1])))
print(encodeAction([1,1]))
print(decodeAction(encodeAction([1,1])))