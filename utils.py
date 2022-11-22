import numpy as np
from sklearn import preprocessing


def ZscoreNormalization(x):
  """Z-score normaliaztion"""
  x = (x - np.mean(x)) / np.std(x)
  return x

def load_data(url):
    beef_data = np.loadtxt(url)
    length = beef_data.shape[0]
    data = []
    target = []

    for i in range(length):
        x = beef_data[i][1:]
        x_label = beef_data[i][0]
        x = np.array(x).reshape(-1, 1)
        x = preprocessing.MaxAbsScaler().fit_transform(x)#归一化
        x = x.reshape(-1)
        data.append(x)
        target.append(x_label)

    # data 为数据集数据;target 为样本标签
    data = np.array(data)
    target = np.array(target)

    return data, target