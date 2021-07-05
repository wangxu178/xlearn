
# import numpy as np
#
import cluster
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot
#
# a = np.array([[1, 0], [0.9, 0], [0.8, 0], [10, 0], [10.1, 0], [9.9, 0]])
#
# k = cluster.KMeans(n_clusters=2, init='random', max_iter=500, tol=1e-10, distances='E_distance', algorithm='classic').fit(a)
# print(k.centre_point)
# print(k.label_l)

import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_blobs
# from matplotlib import pyplot

# data, target = make_blobs(n_samples=100, n_features=2, centers=3)
data, target = make_blobs(n_samples=10000, n_features=2, centers=[[-10, -1, -1, -1], [0, 0, 0, 0], [10, 1, 1, 1], [20, 2, 2, 2], [30, 3, 3, 3], [40, 4, 4, 4], [50, 5, 5, 5]],
                  cluster_std=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], random_state=9)
# data = np.array([[1.0, 0], [0.98, 0], [0.99, 0], [10.0, 0], [10.1, 0], [9.9, 0]])
k = cluster.KMeans(n_init=20, n_clusters=7, init='random', max_iter=500, tol=1e-8, distances='E_distance', algorithm='classic').fit(data)
# K = KMeans(n_clusters=2, random_state=0, init='random', max_iter=200, tol=1e-5).fit(data)
print(k.centre_point)
print(k.label_l)
print(k.score)
print('----------------------------------------------------------')
# print(K.cluster_centers_)
# print(K.labels_)

# 在2D图中绘制样本，每个样本颜色不同
pyplot.scatter(data[:,0],data[:,1]);
pyplot.show()


