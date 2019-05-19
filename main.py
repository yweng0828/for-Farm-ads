#!/usr/bin/env python
# coding: utf-8

# In[2]:


from kNN import handw_KNN
from LDA import handw_LDA1
from LDA import handw_LDA2
from LDA import sklearn_LDA
from SVM import handw_SVM
from SVM import sklearn_SVM
from preprocess import loadData
from preprocess import cross_10folds
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from numpy import *
import time


# In[4]:


# 这里可以选择数据集
# small是小数据集
# farm-ad是整个数据集
fileName = r"data\small" # 测试小数据集
# fileName = r"data\farm-ads" # 测试整个大的数据集

totdata_x, totdata_y = loadData(fileName) # 加载数据

startTime = time.time()

# 下面可以选择多种算法

handw_KNN(totdata_x, totdata_y) # 测试手写KNN算法

# handw_LDA1(totdata_x, totdata_y) # 测试手写的LDA1
# handw_LDA2(totdata_x, totdata_y) # 测试手写的LDA2
# sklearn_LDA(totdata_x, totdata_y) # 测试sklearn中的LDA

# handw_SVM(totdata_x, totdata_y) # 测试手写的SVM算法
# sklearn_SVM(totdata_x, totdata_y) # 测试sklearn中的SVM算法

endTime = time.time()

print('程序运行时间：', endTime-startTime, '秒')


# In[ ]:




