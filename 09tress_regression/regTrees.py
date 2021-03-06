#CART
from numpy import *
import matplotlib.pyplot as plt


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


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2


def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + \
            sum(power(rSet[:,-1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2
        errorMerge = sum(power(testData[:,-1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merge')
            return treeMean
        else:
            print('not merge')
            return tree
    else:
        return tree


def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('The matrix is singular')
    ws = xTx.I*(X.T*Y)
    return ws,X,Y


def modelLeaf(dataSet):
    ws, X, Y =linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X*ws
    return sum(power(Y - yHat, 2))


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)


def treeForceCast(tree, inData,modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForceCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForceCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForceCast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForceCast(tree, mat(testData[i]), modelEval)
    return yHat



# myData = loadDataSet('ex2.txt')
# myMat = mat(myData)
# myTree = createTree(myMat, regLeaf, regErr, (0,1))
# myTestData = loadDataSet('ex2test.txt')
# myTestMat = mat(myTestData)
# pruneTree = prune(myTree, myTestMat)
#
#
# myMat2 = mat(loadDataSet('exp2.txt'))
# myTree2 = createTree(myMat2,modelLeaf, modelErr, (1,10))
# print(myTree2)


trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
bikeTree = createTree(trainMat, modelLeaf, modelErr,(1,20))
yHat = createForceCast(bikeTree, testMat[:,0],modelTreeEval)
print(corrcoef(yHat, testMat[:,1], rowvar=0)[0,1])


fig = plt.figure()
ax = fig.add_subplot(111)
# ax.scatter(myMat2[:,0],myMat2[:,1])
plt.show()
