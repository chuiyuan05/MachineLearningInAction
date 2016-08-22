import numpy as np


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    '''Thus j !=i'''
    j = i
    while (j ==i):
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H ,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMat, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(classLabels).transpose()
    b =0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    my_iter=0
    while (my_iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMatrix).T*\
                        (dataMatrix*dataMatrix[i,:].T))+b
            Ei = fXi - float(labelMatrix[i])
            if labelMatrix[i]*Ei < -toler and alphas[i] < C or \
                labelMatrix[i]*Ei > toler and alphas[i] >0 :
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas,labelMatrix).T*\
                            (dataMatrix*dataMatrix[j,:].T))+b
                Ej = fXj - float(labelMatrix[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if labelMatrix[i] != labelMatrix[j]:
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] -C)
                    H = min(C, alphas[j] +alphas[i])
                if L == H:
                    print('L==H')
                    continue
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T - \
                    dataMatrix[i,:]*dataMatrix[i,:].T - \
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >=0:
                    print('eta >=0')
                    continue
                alphas[j] -= labelMatrix[j]*(Ei -Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alpha_j_old)< 0.001:
                    print('j not moving enough')
                    continue
                alphas[i] += labelMatrix[j]*labelMatrix[i]*\
                             (alpha_j_old - alphas[j])
                b1 = b - Ei - labelMatrix[i]*(alphas[i] - alpha_i_old)*\
                    dataMatrix[i,:]*dataMatrix[i,:].T - \
                    labelMatrix[j]*(alphas[j] - alpha_j_old)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej -labelMatrix[i]*(alphas[i] - alpha_i_old)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMatrix[j]*(alphas[j] - alpha_j_old)*\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if 0 > alphas[i] and C > alphas[i] :
                    b = b1
                elif 0< alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1+b2)/2
                alphaPairsChanged += 1
                print('iter: %d,i %d, pairs changed: %d' %(my_iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
             my_iter = my_iter + 1
        else:
             my_iter = 0
        print('iter number: %d.' % my_iter)
    return b, alphas


class optStruct :
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros(self.m, 2))


def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T*\
                (oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]




dataMatrix, labelMatrix = loadDataSet('./testSet.txt')
b, alphas = smoSimple(dataMatrix, labelMatrix, 0.6, 0.001, 40)
print(alphas[alphas>0])
print(b)





