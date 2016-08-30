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


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(array(dataSet)[:,j])
        rangeJ = float(max(array(dataSet)[:,j]) - minJ)
        centroids[:, j] = minJ + rangeJ*random.rand(k,1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    dataSet = mat(dataSet)
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataMat, k, distMeas = distEclud):
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataMat, axis =0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataMat[j,:])
    while len(centList) < k:
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataMat[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClusterAss = kMeans(ptsInCurrCluster, 2 ,distMeas)
            seeSplit = sum(splitClusterAss[:,1])
            seeNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            if (seeSplit + seeNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClusterAss.copy()
            bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
            bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
            print('the bestCentToSplit is '+str(bestCentToSplit))
            print('the len of bestClusterAss is '+ str(len(bestClustAss)))
            centList[bestCentToSplit] = bestNewCents[0,:]
            centList.append(bestNewCents[1,:])
            #print(centList)
            clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]=bestClustAss
        return mat(centList), clusterAssment



#dataMat = loadDataSet('testSet.txt')
#centMat = randCent(dataMat, 2)
#centroids, clusterAssment = kMeans(dataMat, 4)
dataMat = loadDataSet('testSet2.txt')
centList, newAssment = biKmeans(mat(dataMat), 3)
print(centList)
