import numpy as np


def loadSimpData():
    datMat = np.matrix([[1.,2.1],
                        [2., 1.1],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump( dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}  # dict
    bestClassEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                print('i: %i, thresh: %.2f, thresh inequal: %s, weighted error: %.3f' \
                      %(i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print(D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print('classEst:%f' % classLabels)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        # print('aggClassEst:%f' % aggClassEst)
        aggErrors = np.multiply(np.sign(aggClassEst)!= np.mat(classLabels).T, np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print('total error: %f' %errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr


def adaClassify(data2class, classifierArr):
    dataMatrix = np.mat(data2class)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.ones((m,1)))
    #print(classifierArr)
    # use every weak DT
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


D = np.mat(np.ones((5,1))/5)
datMat, classLabels = loadDataSet('./horseColicTraining2.txt')
# print(datMat)
# print(classLabels)
# print(buildStump(datMat, classLabels, D))
classifierArr = adaBoostTrainDS(datMat, classLabels, 100)

testMat, testLabelMat = loadDataSet('./horseColicTest2.txt')
prediction10 = adaClassify(testMat, classifierArr)
errArr = np.mat(np.ones((67,1)))
errSum = errArr[prediction10 != np.mat(testLabelMat).T].sum()
print('predict error rate: %f' % float(errSum/67))

