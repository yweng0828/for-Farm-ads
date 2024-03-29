#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from preprocess import loadData
from preprocess import cross_10folds
from numpy import *
from sklearn import svm

# In[ ]:


def sklearn_SVM(totdata_x, totdata_y):
    res = 0.0
    for j in range(0, 10):
        train_x, train_y, test_x, test_y = cross_10folds(totdata_x, totdata_y, j)
        model = svm.SVC(kernel='linear', C=1, gamma=1)
        model.fit(train_x, train_y)
        
        right = 0
        for i in range(test_x.shape[0]):
            if(model.predict([test_x[i]])==test_y[i]):
                right = right + 1
        
        res += right/test_y.shape[0]
        print("第 %d 次的准确率为 %f" %(j, right/test_y.shape[0]))
    
    print("最后的准确率为 %f" %(res/10))


# In[ ]:


def selectJrand(i, m):
    j=i;
    while(j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj, H, L):
    if(aj>H):
        aj = H
    if(L>aj):
        aj = L
    return aj


# In[ ]:


def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = X.shape
    K = mat(zeros((m,1)))
    if (kTup[0]=='lin'): K = X * A.T   #linear kernel
    elif (kTup[0]=='rbf'):
        for j in range(m):
            deltaRow = X[j,:] - A
            
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: 
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


# In[ ]:


class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = dataMatIn.shape[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
        
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList) > 1):
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if (k == i): continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0))):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if (L==H): 
            #print ("L==H"); 
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if (eta >= 0): 
            #print ("eta>=0"); 
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): 
            #print ("j not moving enough"); 
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if ((0 < oS.alphas[i]) and (oS.C > oS.alphas[i])): oS.b = b1
        elif ((0 < oS.alphas[j]) and (oS.C > oS.alphas[j])): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0


# In[ ]:


def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iters = 0
    entireSet = True; alphaPairsChanged = 0
    while ((iters < maxIter) and ((alphaPairsChanged > 0) or (entireSet))):
        alphaPairsChanged = 0
        if (entireSet):   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                #print ("fullSet, iter: %d i:%d, pairs changed %d" % (iters,i,alphaPairsChanged))
            iters += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                #print ("non-bound, iter: %d i:%d, pairs changed %d" % (iters,i,alphaPairsChanged))
            iters += 1
        if (entireSet): entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        #print ("iteration number: %d" % iters)
    return oS.b,oS.alphas


# In[ ]:


def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = X.shape
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w


# In[ ]:


def test(train_x, train_y, test_x, test_y, k1=1.3):
    
    b,alphas = smoP(train_x, train_y, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(train_x); labelMat = mat(train_y).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    # print ("there are %d Support Vectors" % sVs.shape[0])
    m,n = datMat.shape
    errorCount = 0

    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if (sign(predict)!=sign(train_y[i])): errorCount += 1
    #print ("the training error rate is: %f" % (float(errorCount)/m))
    
    errorCount = 0
    datMat=mat(test_x); labelMat = mat(test_y).transpose()
    m,n = datMat.shape
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if (sign(predict)!=sign(test_y[i])): errorCount += 1    
    #print ("the test error rate is: %f" % (float(errorCount)/m))
    return (float(errorCount)/m)


# In[ ]:


def handw_SVM(totdata_x, totdata_y):
    res = 0.0
    for j in range(0, 10):
        train_x, train_y, test_x, test_y = cross_10folds(totdata_x, totdata_y, j)
        #print(train_x.shape)
        #print(train_y.shape)
        #print(test_x.shape)
        #print(test_y.shape)
        temp = test(train_x, train_y, test_x, test_y)
        res = res + temp
        print("第 %d 次的准确率为 %f" %(j, temp))
    
    print("最后的准确率为 %f" %(res/10))


# In[ ]:




