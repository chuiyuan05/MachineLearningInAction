from numpy import *

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))-1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print('This matrix is not singular, can not do reverse')
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)
    if linalg.det(xTx) == 0.0:
        print('this matrix is singular, can not do inverse')
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws


def lwlrTest(testArr, xArr, yArr, k =1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr, yArr,k)
    return yHat


xArr, yArr = loadDataSet('./ex0.txt')
ws = standRegres(xArr, yArr)
print(ws)
yHat = lwlrTest(xArr, xArr, yArr, 0.02)

xMat = mat(xArr)
xCopy = xMat.copy()
yMat = mat(yArr)
# yHat = xMat*ws
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# ax.scatter(xMat[:,1].flatten(), yMat.T[:,0].flatten())



srcIndex = xMat[:,1].argsort(0)
xSorted = xMat[srcIndex][:,0,:]
ax.scatter(xMat[:,1], yHat)
plt.show()




