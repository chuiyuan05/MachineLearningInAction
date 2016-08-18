#!/usr/bin/python

import numpy as np
from os import listdir

def classify0(inX ,dataSet, labels, k ):
    dataSetSize = np.shape(dataSet)[0]
    diffMat = dataSet - np.tile(inX,(dataSetSize,1))
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5
    sortedDistIndex = distances.argsort()
    classCount ={}
    for i in range(k):
        voteLable = labels[sortedDistIndex[i]]
        classCount[voteLable] = classCount.get(voteLable, 0)+1
    sortedClassCount = sorted(classCount.__iter__(),
                              key = lambda d:d[0], reverse = True)
    return sortedClassCount[0][0]


def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(line[j])
    return returnVect

def handwriting_class_test():
    hw_labels = []
    trainingFileList = listdir('./digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumberStr = fileStr.split('_')[0]
        hw_labels.append(classNumberStr)
        trainingMat[i,:]=img2vector('./digits/trainingDigits/%s' %fileNameStr)

    testFileList = listdir('./digits/testDigits')
    errorCount =0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumberStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./digits/testDigits/%s' %fileNameStr)
        classifyResult = classify0(vectorUnderTest,
                                  trainingMat, hw_labels, 1)
        print('classify result is %s, the real answer is %s.' %(classifyResult,
                                                                classNumberStr))
        if int(classNumberStr) != int(classifyResult):
            errorCount +=1
    print('\n the total errorcount is %d ' % errorCount)
    print('\n the error rate is %f' %(errorCount/float(mTest)))

handwriting_class_test()
