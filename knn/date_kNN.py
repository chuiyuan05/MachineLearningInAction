#!/usr/bin/python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index =0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(str(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def auto_norm(dataSet):
    '''normValue = (oldValue-min)/(max -min)'''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = np.shape(dataSet)[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges ,(m,1))
    return normDataSet, ranges, minVals


def classify0(inX, dataSet, labels, k):
    dataSetSize = np.shape(dataSet)[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDisIndex = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlable = labels[sortedDisIndex[i]]
        classCount[voteIlable] = classCount.get(voteIlable,0)+1
    print(classCount)
    sortedClassCount = sorted(classCount.__iter__(),
                              key = lambda d:d[0],reverse = True)
    return sortedClassCount[0][0]


# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# normMat, ranges, minVals = auto_norm(datingDataMat)
# print(normMat)
# print(datingLabels)
# classifierResult = classify0(normMat[1,:], normMat[100:1000,:],
#                            datingLabels[100:1000], 3)

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = auto_norm(datingDataMat)
    m = np.shape(normMat)[0]
    numTestVecs = int(hoRatio*m)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
                           datingLabels[numTestVecs:m], 3)
        print('the classify result is %s, the real answer is %s' % (classifierResult, datingLabels[i]))
        if int(classifierResult) != int(datingLabels[i]):
            errorCount += 1

    print('the total error rate is %f' % (errorCount/float(numTestVecs)))


def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# group, labels = createDataSet()
# sortedClassCount = classify0([0,0], group, labels, 3)
# print(sortedClassCount)
datingClassTest()
# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# datingDataMat, ranges, minVals = auto_norm(datingDataMat)
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(datingDataMat[:,0],datingDataMat[:,1],
#            15.0*np.array(datingLabels),15.0*np.array(datingLabels))
# plt.show()



