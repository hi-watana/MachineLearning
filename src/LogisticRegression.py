#!/usr/bin/env python3
import numpy as np
import numpy.linalg as LA

GRAD = "gradient"
NEWTON = "newton"

class LogisticRegression:
    def __init__(self, y, X, method=GRAD, standardization=True, tmax=100, lam=1):
        self.y = y
        self.n, self.dim = X.shape
        self.mu = X.sum(0) / self.n
        if standardization:
            self.C = np.linalg.cholesky(sum(xi.T * xi for xi in X) / self.n - self.mu.T * self.mu)
            self.X = (X - self.mu) * self.C.I # standardization
        else:
            self.X = X

        self.w = np.matrix([float() for _ in range(self.dim - 1)] + [1.0])
        self.lam = lam # lambda
        self.history = self.w.tolist() # sequence of w
        if method == GRAD:
            self.getD = self.lossFunctionPrime
            self.alpha = 1.0 / float(max(LA.eig(X * X.T + 2 * self.lam * np.identity(self.n))[0]) / 4.0)
        else:
            self.getD = lambda: self.lossFunctionPrime() * self.lossFunctionDoublePrime().I
            self.alpha = 1
        self.d = self.getD()

        self.epsilon1 = 1e-4
        self.epsilon2 = 1e-18
        if tmax < 0:
            self.solve = self.__unlimitedSolve
            self.tmax = -1
            self.rho = 0.99
        else:
            self.solve = self.__limitedSolve
            self.tmax = tmax


    def lossFunction(self, w): # J(w)
        return (sum(map(lambda yi, xi: np.log(1 + np.exp(-yi * w * xi.T)), self.y, self.X)) + self.lam * w * w.T).tolist()[0][0]

    def lossFunctionPrime(self): # dJ(w)/dw
        return 2 * self.lam * self.w - sum(map(lambda yi, xi: (1 - 1 / (1 + np.exp(-yi * self.w * xi.T))) * yi * xi, self.y, self.X))

    def lossFunctionDoublePrime(self): # d^2J(w)/dw^2
        def __f(yi, xi):
            z = yi * self.w * xi.T
            return (xi.T * xi) * np.exp(- z).A1[0] / ((1 + np.exp(- z)) ** 2)
        return 2 * self.lam * np.identity(self.dim) + sum(map(__f, self.y, self.X))

    def update(self):
        self.w -= self.alpha * self.getD()
        self.w /= np.sqrt(self.w * self.w.T)

    def __limitedSolve(self):
        for _ in range(self.tmax):
            self.update()
            self.history += self.w.tolist()

    def __unlimitedSolve(self):
        while True:
            self.update()
            self.alpha *= self.rho
            d = self.lossFunctionPrime()
            self.history += self.w.tolist()
            dd = d - self.d
            if d * d.T < self.epsilon1 or dd * dd.T < self.epsilon2:
                break
            else:
                self.d = d
                print(d)

    def predict(self, X):
        return np.matrix([[1] if 1.0 / (1 + np.exp(- self.w * xi.T)) >= 0.5 else [-1] for xi in X])


