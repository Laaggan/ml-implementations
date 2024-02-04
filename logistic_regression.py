import numpy as np

class LogisticRegression():
    def logistic_function(self, x):
        # Logistic function (sigmoid)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        intercept = np.log(np.mean(y)/(1 - np.mean(y))) # intercept
        beta = np.concatenate((np.array([intercept]).astype(np.float64), np.zeros(X.shape[1] - 1).astype(np.float64)), axis=0)
        #These three below is probably better as parameters to function
        eta = 0.5
        tolerance = 1e-12
        curr_tol = 1
        
        it = 0
        iter_max = 100
        ll = 0

        while curr_tol > tolerance and it < iter_max:
            it += 1
            ll_old = ll

            mu = self.logistic_function(np.dot(beta.transpose(), X.transpose()))
            g = np.dot(X.transpose(), mu - y)
            S = np.diag(mu * (1 - mu))
            H = np.dot(np.dot(X.transpose(), S), X)
            beta = beta - eta*np.dot(np.linalg.inv(H), g)

            ll = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
            curr_tol = np.abs(ll - ll_old)
        
        return {
            'beta': beta,
            'iter': it,
            'tol': curr_tol,
            'loglik': ll
        }



