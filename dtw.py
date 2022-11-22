from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
import numpy as np


def dtw(x, y, dist=lambda x, y: np.abs(x - y), warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure 用作成本度量的距离
    :param int warp: how many shifts are computed. 计算了多少班次
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|. 窗口大小限制匹配条目索引之间的最大距离|i，j|
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    在路径的非对角移动上应用的权重。随着s变大，扭曲路径越来越偏向对角线
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    最小距离、成本矩阵、累积成本矩阵和包裹路径
    """

    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0

    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
        # print(D0, 'D0-1')
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
        # print(D0,'D0-1 初始化累计距离矩阵')
    D1 = D0[1:, 1:]  # view
    # print(D1, 'D1-1')
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    # print(D1, 'D1-2 初始化距离矩阵')
    C = D1.copy()
    # print(C, 'C')
    jrange = range(c)
    # print(jrange, 'jrange-0 列数')
    # print(D0,'D0-2 赋值距离后但还未累计的累计距离矩阵')
    for i in range(r):   # i 行数
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:  # j 列数
            min_list = [D0[i, j]]
            # # print(min_list, 'min_list-j')
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]

                # print(min_list, 'min_list-k')
            D1[i, j] += min(min_list)
            # print(min_list.index(min(min_list)))
    # print(D1, 'D1-3 累计距离矩阵')
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    # return D1[-1, -1], C, D1, path
    return D1[-1, -1]



def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    # print(D)
    # print(i, j, 'i, j')
    # print(p, q, 'p, q')
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    # print(array(p), 'array(p)')
    # print(array(q), 'array(q)')
    return array(p), array(q)


