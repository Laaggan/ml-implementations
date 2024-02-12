from math import sqrt
import numpy as np

def euclidean_distance(x1, x2):
    diff = x1 - x2
    return sqrt(sum(map(lambda x: x**2, diff)))

def print_pythagorean_triples_up_to_n(n):
    for i in range(1, n):
        for j in range(i, n):
            x1 = np.array([0, 0])
            x2 = np.array([i, j])
            result = euclidean_distance(x1, x2)
            if not abs(round(result) - result) > 0:
                print(i, j, result)


class KnnClassifier:
    def __init__(self, K):
        self.K = K

    def fit(self, X):
        pass

    def predict(self, X, Y, x):
        result = np.zeros(X.shape[0])
        for i, x_i in enumerate(X):
            result[i] = euclidean_distance(x_i, x)
        
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
            
