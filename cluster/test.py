import numpy as np
import random
import copy

# a = np.array([[1,2,3],[4,5,6]])
# b = [1,2,3]
# x = b.index(1)
# print(x)
# print(a.shape())
# b = np.random.randint(-1000, 1000, size=(5, 3))
# a = b.astype(np.float32)
# print(len(a))

data = [[1],[2],[3],[4],[5]]

s = copy.copy(data)
print(s)
s[0] = [8]
print(data)
print(s)