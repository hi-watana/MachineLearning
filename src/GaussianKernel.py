import numpy as np

class GaussianKernel:
    def __init__(self, X, alpha=5, dim=2):
        self.X = X
        self.n = X.shape[0]
        self.alpha = alpha
        self.dim = dim if dim < self.n else self.n - 1
        self.zk = list(range(self.n))[:self.dim]
        np.random.seed(1)
        np.random.shuffle(self.zk)

    def phiX(self, X):
        return np.matrix(np.matrix([[np.exp(- self.alpha * (x - self.X[zk]) * (x - self.X[zk]).T).A1[0] for zk in self.zk] for x in X]))
