#!/usr/bin/env python
# coding: utf-8

# In[181]:


import re
import numpy as np
import operator


# In[182]:


# 查看'ad-' 'title-' 类型的数据有多少种
# 发现只有{'ad', 'header', 'title'}
def findAllPre(fileName):
    fr = open(fileName)
    preSet = set() # 存储‘ad-’等的前缀，例如ad，header
    
    # 查看'ad-' 'title-' 类型的数据有多少种
    for line in fr.readlines():
        curLine = line.strip().split()
        for curStr in curLine:
            pos = curStr.find('-')
            if(pos!=-1 and pos>0):
                preSet.add(curStr[0:pos])
                
    return preSet # 返回所有可能的前缀


# In[183]:


# 找出所有单词
def findAllWord(fileName, uselessSet):
    fr = open(fileName)
    totWordDict = dict()
    cnt = 0
    for line in fr.readlines():
        curLine = re.split("-| ", line)
        for curStr in curLine:
            if(curStr not in uselessSet and curStr != "" and curStr != "1"):
                if(totWordDict.get(curStr, -1)==-1):
                    totWordDict[curStr] = cnt
                    cnt = cnt+1

    return totWordDict


# In[184]:


# 生成数据矩阵
def genDataMatrix(fileName, dataNum, uselessSet, totWordDict):
    totdata_x = np.zeros([dataNum, len(totWordDict)])
    totdata_y = np.zeros(dataNum)
    
    fr = open(fileName)
    for line,i in zip(fr.readlines(), range(0,dataNum)):
        totdata_y[i] = line.strip().split()[0]  # 第一列是结果 1 和 -1 
        curLine = re.split("-| ", line)
        for curStr in curLine:
            if(curStr not in uselessSet and curStr != "" and curStr != "1"):
                totdata_x[i][totWordDict[curStr]] = 1
    return totdata_x, totdata_y


# In[185]:


def cross_10folds(totdata_x, totdata_y, choidx):
    dataNum = totdata_x.shape[0]
    perFold = dataNum / 10
    startidx = int(choidx * perFold)
    endidx = int((choidx+1)*perFold)
    
    test_x = totdata_x[startidx:endidx]
    test_y = totdata_y[startidx:endidx]
    
    train_x = np.delete(totdata_x, range(startidx, endidx), axis=0)
    train_y = np.delete(totdata_y, range(startidx, endidx), axis=0)
    
    return train_x, train_y, test_x, test_y


# In[186]:


# fileName是文件路径和文件名 
# choidx是选择了哪一批数据作为测试集，范围从0-9
def loadData(fileName):
    dataNum = len(open(fileName,'r').readlines()) # 获取总行数
    fr = open(fileName)

    uselessSet = findAllPre(fileName) # 不想要的前缀
    # print(uselessSet)
    uselessSet.add("page") # 加上两个不要的后缀
    uselessSet.add("found")
    uselessSet.add("com")
    uselessSet.add("www")
    print("需要剔除的数据：", uselessSet)

    totWordDict = findAllWord(fileName, uselessSet) # 所有单词的集合

    print("totword number= ", len(totWordDict))


    totdata_x, totdata_y = genDataMatrix(fileName, dataNum, uselessSet, totWordDict)
    # print(totdata) 一个非常稀疏的矩阵
    # print(totdata.shape)
    
    totdata_x = totdata_x.astype('float32')
    totdata_y = totdata_y.astype('int32')
    
    return totdata_x, totdata_y


# In[187]:


def kNNclassify(inX, dataSet, labels, k=5):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1))- dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# In[188]:





# In[ ]:




