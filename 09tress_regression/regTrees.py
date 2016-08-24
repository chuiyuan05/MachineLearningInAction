#CART
from numpy import *


def loadDataSet(filename):
    numRow = len(open(filename).readline().split('\t'))
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numRow):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    mat0Index = nonzero(dataSet[:,feature] > value)[0]
    mat1Index = nonzero(dataSet[:,feature] <= value)[0]
    mat0 = dataSet[mat0Index,:]
    mat1 = dataSet[mat1Index,:]
    return mat0, mat1


# testMat = mat(eye(4))
# print(testMat)
# mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
# print(mat0)
# print(mat1)



def createTree(dataSet, leafType, errType , ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None :
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def regLeaf(dataSet):
    return mean(dataSet[:,-1])


def regErr(dataSet):
    return var(dataSet[:,-1])*shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType, errType, ops):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) ==1:
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue =0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].flat):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if shape(mat0)[0]< tolN or shape(mat1)[0] < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS)< tolS:
        return None, leafType(dataSet)
    mat0, mat1 =binSplitDataSet(dataSet, bestIndex, bestValue)
    if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue


myData = loadDataSet('ex00.txt')
myMat = mat(myData)
myTree = createTree(myMat, regLeaf, regErr, (1,4))
print(myTree)

