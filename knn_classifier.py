from math import sqrt
import numpy as np

def manhattan_distance(x1, x2):
    return minowski_distance(x1, x2, 1)

def euclidean_distance(x1, x2):
    return minowski_distance(x1, x2, 2)

def minowski_distance(x1, x2, p):
    diff = x1 - x2
    return (sum(map(lambda x: x**p, diff)))**(1/p)

def print_pythagorean_triples_up_to_n(n, p=2):
    for i in range(1, n):
        for j in range(i, n):
            x1 = np.array([0, 0])
            x2 = np.array([i, j])
            result = minowski_distance(x1, x2, p)
            if not abs(round(result) - result) > 0:
                print(i, j, result)


class KnnClassifier:
    def __init__(self, K, p=2):
        self.K = K
        self.p = p

    def fit(self, X):
        pass

    def predict(self, X, Y, x):
        result = np.zeros(X.shape[0])
        for i, x_i in enumerate(X):
            result[i] = minowski_distance(x_i, x, self.p)
        
        sort_index = np.argsort(result)
        knn_index = sort_index[:self.K]
        nn = Y[knn_index]
        nn_unique = np.unique(nn)
        
        y_max = 0
        y_value = None
        for y in nn_unique:
            y_count = (nn == y).sum()
            if y_count > y_max:
                y_max = y_count
                y_value = y
        
        return y_value
            
