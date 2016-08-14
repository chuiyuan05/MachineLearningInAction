#!/usr/bin/python

from numpy import *
import operator


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


def classify0(inX, dataSet, labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() # index small -> big
    classCount ={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    return classCount

group, labels = createDataSet()
classCount = classify0([0,0], group, labels, 3)
print(classCount)

sortedClassCount = sorted(classCount.__iter__(),
                          key=lambda d:d[0], reverse = True)
print(sortedClassCount)