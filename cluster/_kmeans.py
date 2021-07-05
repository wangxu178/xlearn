# -*- coding: utf-8 -*-
"""
@Time: 2021.07.02
@Author : wangxu
@Emil : 1440197502@qq.com
@Description : K-means cluster
"""
import numpy as np
import logging
import random
import copy
# import _kmeans_c
from ctypes import *


class KMeans:
    """
    algorithm : 1.classic 2.elkan

    init : 1.random 2.k-means++

    distances : 1.E_distance 2.M_distance 3.C_distance

    """

    def __init__(self, n_clusters=8, init='random', n_init=500,
                 max_iter=500, tol=1e-8, distances='E_distance',
                 n_thread=10, algorithm='classic'):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.distances = distances
        self.n_init = n_init
        self.n_thread = n_thread
        self.algorithm = algorithm
        self.centre_point = np.ndarray([])
        self.label_l = np.ndarray([])
        self.score = -2

    def fit(self, x):
        iter_c = 0
        if type(x) is np.ndarray:
         tmp_cp = np.ndarray([])
         tmp_l = np.ndarray([])
         n_ = self.n_init
         tmp_ss = -2
         while(n_):
            _row, _col = x.shape
            if self.init == 'random':
                init_array_int = random.sample(list(x), self.n_clusters)
                # init_array_int = np.random.randint(-10, 10, size=(self.n_clusters, _col))
                init_array_float32 = np.array(init_array_int, dtype=np.float32)
            else:
                pass
            if self.algorithm == 'classic' and self.distances == 'E_distance':
             while(1):
                # Todo:1.to C----------------------------------------------------------------
                iter_c = iter_c + 1
                label_list = []
                for i in x:
                    min_ = -1
                    label_ = -1
                    for j in range(len(init_array_float32)):
                        s_ = 0
                        for k in range(len(i)):
                            s_ = np.square(i[k]-init_array_float32[j][k]) + s_
                        q_ = np.sqrt(s_)
                        if min_ == -1:
                            min_ = q_
                            label_ = 0
                        elif q_ < min_:
                            min_ = q_
                            label_ = j
                    label_list.append(label_)
                new_centre = []
                for i in range(len(init_array_float32)):
                    sum_ = []
                    c_label = 0
                    for j in range(len(label_list)):
                        if label_list[j] == i:
                            c_label = c_label + 1
                            if len(sum_) == 0:
                                sum_ = copy.copy(x[j])
                            else:
                                for k in range(len(sum_)):
                                    sum_[k] = sum_[k] + x[j][k]
                        else:
                            continue
                    new_centre_ = np.array(sum_)/c_label
                    new_centre.append(new_centre_)
                init_array_float32 = new_centre
                if iter_c > self.max_iter:
                    tmp_cp = np.array(new_centre)
                    tmp_l = np.array(label_list)
                    break
                sum___ = 0
                for i in range(len(new_centre)):
                    sum__ = 0
                    for j in range(len(new_centre[i])):
                        sum__ = np.square(new_centre[i][j] - init_array_float32[i][j]) + sum__
                    sum___ = np.sqrt(sum__) + sum___
                if sum___ < self.tol:
                    tmp_cp = np.array(new_centre)
                    tmp_l = np.array(label_list)
                    break
                # Todo:1-----------------------------------------------------------------
            else:
                pass
            s = 0
            # Todo:2.to C----------------------------------------------------------------
            for i in range(len(tmp_cp)):
                s_sum_cp = 0
                for j in range(len(tmp_cp)):
                    sum_cp = 0
                    if i != j:
                        for k in range(len(tmp_cp[j])):
                            sum_cp = np.square(tmp_cp[i][k] - tmp_cp[j][k]) + sum_cp
                        s_sum_cp = np.sqrt(sum_cp) + s_sum_cp
                s_sum_cp_m = s_sum_cp/(len(tmp_cp)-1)
                s_sum_d = 0
                cc = 0
                for k in range(len(tmp_l)):
                    sum_d = 0
                    if tmp_l[k] == i:
                        cc = cc + 1
                        for m in range(len(x[k])):
                            sum_d = sum_d + np.square(tmp_cp[i][m] - x[k][m])
                        s_sum_d = np.sqrt(sum_d) + s_sum_d
                s_sum_d_m = s_sum_d/cc
                s = (s_sum_cp_m - s_sum_d_m) / max([s_sum_cp_m, s_sum_d_m]) + s
            tmp_s = s/len(tmp_cp)
            # Todo:2.--------------------------------------------------------------------
            if tmp_ss == -2:
                tmp_ss = tmp_s
                self.centre_point = tmp_cp
                self.label_l = tmp_l
                self.score = tmp_s
            elif tmp_ss < tmp_s:
                tmp_ss = tmp_s
                self.centre_point = tmp_cp
                self.label_l = tmp_l
                self.score = tmp_s
            n_ = n_ - 1
         return self
        else:
            logging.warning('---Please input the parameter with numpy.ndarray data type---')

    def predict(self, y):
        if type(y) is np.ndarray:
            label_l = []
            for i in range(len(y)):
                tmp_d = 0
                tmp_l = -1
                for k in range(len(self.centre_point)):
                    sum_y = 0
                    for j in range(len(y[i])):
                        sum_y = sum_y + np.square(y[i][j] - self.centre_point[k][j])
                    s_sum_y = np.sqrt(sum_y)
                    if tmp_l == -1:
                        tmp_l = k
                        tmp_d = s_sum_y
                    elif tmp_d > s_sum_y:
                        tmp_l = k
                        tmp_d = s_sum_y
                label_l.append(tmp_l)
            return label_l
        else:
            logging.warning('---Please input the parameter with numpy.ndarray data type---')
        pass

    def score(self):
        return self.score


