内容：
使用KNN, LDA（包含手写的和sklearn版本）, SVM（包含手写的和sklearn版本）解决Farm-ads数据集(source: http://archive.ics.uci.edu/ml/datasets/Farm+Ads)


实验环境：
本实验使用的jupyter notebook作为IDE

__pycache__、.ipynb_checkpoints：均为自动生成文件

/data：
farm-ads：整个数据集
farm-ads-vect：整个数据集
small：数据集的部分
farm-data.zip：源数据


程序入口：（高度封装了以下文件）
main.ipynb
main.py


预处理文件：（包括数据读入，数据清洗，10折操作）
preprocess.ipynb
preprocess.py

KNN实现文件：（包含手写KNN函数）
KNN.ipynb
KNN.py

LDA实现文件：（包含手写LDA函数LDA1和LDA2，以及使用sklearn的LDA）
LDA.ipynb
LDA.py

SVM实现文件：（包含手写SVM和使用sklearn的SVM）
SVM.ipynb
SVM.py

