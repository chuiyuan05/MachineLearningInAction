import numpy as np


def loadDataSet():
    postingList = [['my', 'dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','lickes','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]

    classVect = [0,1,0,1,0,1] #class of postingList[i]
    return postingList, classVect


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inSet):
    retVect = [0]*len(vocabList)
    for word in inSet:
        if word in vocabList:
            retVect[vocabList.index(word)] += 1 # or =1
        else:
            print('the word: %s is not in vocabList' % word)
    return retVect


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p1Vect, p0Vect, pAbusive


# postingList, classVect = loadDataSet()
# vocabList = createVocabList(postingList)
# trainMat = []
# for postingDoc in postingList:
#     trainMat.append(setOfWords2Vec(vocabList,postingDoc))
# p0, p1, pAb = trainNB0(trainMat, classVect)
# print(sum(p1))


def classifyNB(vect2classify, p0Vect, p1Vect, pClass1):
    p1 = sum(vect2classify*p1Vect) + np.log(pClass1)
    p0 = sum(vect2classify*p0Vect) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOfPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    trainMat = []
    for postingDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postingDoc))
    p1Vect , p0Vect, pA = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    testDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(classifyNB(testDoc, p0Vect, p1Vect, pA))
    testEntry = ['stupid', 'garbage']
    testDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(classifyNB(testDoc, p0Vect, p1Vect, pA))


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        print(i)
        wordList = textParse(open('./email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('./email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p1V,p0V,pSpam = trainNB0(np.array(trainMat), np.array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V,p1V,pSpam)!= classList[docIndex]:
            errorCount += 1
    print('the error rate is : %f' % (float(errorCount)/len(testSet)))


spamTest()