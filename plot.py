import matplotlib.pyplot as plt
import numpy as np
from utils import *
#设定画布。dpi越大图越清晰，绘图时间越久
dataset_name = 'Fungi'
#导入数据
X_train, y_train = load_data(r'C:\Users\th\Desktop\%s\%s_TRAIN.txt'% (dataset_name, dataset_name))
X_test, y_test = load_data(r'C:\Users\th\Desktop\%s\%s_TEST.txt'% (dataset_name, dataset_name))


for i in range(X_train.shape[0]):
    fig = plt.figure(figsize=(10, 6), dpi=300)
    x = list(np.arange(1, X_train.shape[1] + 1))
    y = list(X_train[i])
    # 绘图命令
    plt.title('%s_TRAIN' %dataset_name)
    plt.plot(x, y, 'b', linewidth=0.5)
    plt.plot(x, y, 'r.', markersize=2)
    #show出图形
    # plt.show()
    #保存图片
    fig.savefig(r"C:\Users\th\Desktop\%s\image\%s.png" % (dataset_name, i))

