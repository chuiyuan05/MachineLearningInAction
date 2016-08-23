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


xArr, yArr = loadDataSet('./ex0.txt')
ws = standRegres(xArr, yArr)
print(ws)

xMat = mat(xArr)
xCopy = xMat.copy()
yMat = mat(yArr)
yHat = xMat*ws
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(xMat[:,1].flatten(), yMat.T[:,0].flatten())

#xCopy.sort(0)
print(xCopy)
xHat = xCopy*ws
ax.plot(xCopy[:,1], yHat)
plt.show()





