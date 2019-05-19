#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from preprocess import loadData
from preprocess import cross_10folds

# In[2]:


def func(x, w):
    return np.dot((x), w)


# In[3]:


# 最标准的写法
def handw_LDA1(X, y):
    X1 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
    X2 = np.array([X[i] for i in range(len(X)) if y[i] == 1])
 
    len1 = len(X1)
    len2 = len(X2)
 
    mju1 = np.mean(X1, axis=0)#求中心点
    mju2 = np.mean(X2, axis=0)
    
    cov1 = np.dot((X1 - mju1).T, (X1 - mju1))
    cov2 = np.dot((X2 - mju2).T, (X2 - mju2))
    Sw = cov1 + cov2
    w = np.dot(np.mat(Sw).I, (mju1 - mju2).reshape((len(mju1),1)))# 计算w
    return w


# In[4]:


def calculate_covariance_matrix(X, Y=np.empty((0,0))):
    if not Y.any():
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)


def handw_LDA2(train_x, train_y):
    x1 = np.array([train_x[i] for i in range(train_x.shape[0]) if train_y[i] == 1])
    x2 = np.array([train_x[i] for i in range(train_x.shape[0]) if train_y[i] == -1])
    
    print(x1.shape)
    print(x2.shape)
    
    # 计算两个子集的协方差矩阵
    S1 = calculate_covariance_matrix(x1)
    S2 = calculate_covariance_matrix(x2)
    Sw = S1 + S2
    
    # 计算两个子集的均值
    mu1 = x1.mean(axis=0)
    mu2 = x2.mean(axis=0)
    mean_diff = np.atleast_1d(mu1 - mu2)
    mean_diff = mean_diff.reshape(train_x.shape[1], -1)
    
    w = np.linalg.pinv(Sw).dot(mean_diff)
    return w


# In[5]:


def sklearn_LDA(totdata_x, totdata_y):
    res = 0.0
    for j in range(0, 10):
        train_x, train_y, test_x, test_y = cross_10folds(totdata_x, totdata_y, j)
        clf = LinearDiscriminantAnalysis()
        clf.fit(train_x, train_y)
        LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)
        right = 0
        for i in range(0, test_x.shape[0]):
            if(clf.predict([test_x[i]])==test_y[i]):
                right = right+1
    
        res += right/test_y.shape[0]
        print("第 %d 次的准确率为 %f" %(j, right/test_y.shape[0]))
    
    print("最后的准确率为 %f" %(res/10))


# In[ ]:




