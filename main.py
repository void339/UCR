import numpy as np
# 导入数据集
from sklearn.neighbors import KNeighborsClassifier

from utils import *
from dtw import *
dataset_name = 'Fungi'
X_train, y_train = load_data(r'C:\Users\th\Desktop\%s\%s_TRAIN.txt'% (dataset_name, dataset_name))
X_test, y_test = load_data(r'C:\Users\th\Desktop\%s\%s_TEST.txt'% (dataset_name, dataset_name))

def ed(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

#构建knn分类模型，并指定 k 值
KNN = KNeighborsClassifier(n_neighbors=1)
#使用训练集训练模型
KNN.fit(X_train, y_train)
#评估模型的得分
score = KNN.score(X_test, y_test)
print(1-score)