import numpy as np

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('./testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inZ):
    return 1.0/(1+np.exp(-inZ))


def gradAscent(dataMatIn, labels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(labels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights


dataMat,labelMat = loadDataSet()
# weigths = gradAscent(dataMat, labelMat)


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xcord1,ycord1,s=30, c='red', marker = 's')
    ax.scatter(xcord2, ycord2,s=30, c='green')
    x = range(-3, 3,1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x ,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLables):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h= sigmoid(sum(dataMatrix[i]*weights))
        error = classLables[i] -h
        weights += alpha*error*dataMatrix[i]
        print(weights)
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+i+j)+0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights += alpha*error*dataMatrix[randIndex]
            del dataIndex[randIndex]
    return weights


weights = stocGradAscent1(np.array(dataMat),labelMat)
print(weights)
plotBestFit(weights)