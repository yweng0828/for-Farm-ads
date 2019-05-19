#!/usr/bin/env python
# coding: utf-8

# In[3]:

import preprocess
from preprocess import cross_10folds
from preprocess import kNNclassify


# In[4]:


# 使用10折交叉验证
def handw_KNN(totdata_x, totdata_y):
    res = 0.0
    for j in range(0, 10):
        train_x, train_y, test_x, test_y = cross_10folds(totdata_x, totdata_y, j)
        #print(train_x.shape)
        #print(train_y.shape)
        #print(test_x.shape)
        #print(test_y.shape)
        
        right = 0
        for i in range(test_x.shape[0]):
            if(kNNclassify(test_x[i], train_x, train_y, 5)==test_y[i]):
                right = right + 1
        
        res += right/test_y.shape[0]
        print("第 %d 次的准确率为 %f" %(j, right/test_y.shape[0]))
    
    print("最后的准确率为 %f" %(res/10))


# In[ ]:




