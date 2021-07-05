cdef extern from "kmeans_c.h":
    int train_c(int a, int b)
    cdef struct p:
      int point[2][2]
      int label[2][2]
    cdef p predict_c()


def train(int a, int b):
    return train_c(a, b)

def predict():
    return predict_c()