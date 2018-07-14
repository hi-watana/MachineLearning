import numpy as np
import numpy.linalg as LA

PG = "PG"
APG = "APG"

class Lasso:
    def __init__(self, mu, A, lam, method=APG, tmax=100):
        self.w = np.matrix([[3, -1]])
        self.mu = mu
        self.A = A
        self.lam = lam
        self.eta = 1.0 / float(max(LA.eig(2 * self.A)[0]))
        self.history = [self.w] # sequence of J(w)
        self.tmax = tmax
        if method == APG:
            self.prevW = self.w
            self.s = 0
            self.update = self.updateAPG
        else:
            self.update = self.updatePG


    def __phi(self, w):
        d = w - self.mu
        return (d * self.A * d.T).tolist()[0][0]

    def lossFunction(self, w):
        return self.__phi(w) + self.lam * np.abs(w).sum()

    def __phiPrime(self, w):
        d = w - self.mu
        return 2 * d * self.A

    def prox_eta(self, mu):
        etalambda = self.eta * self.lam
        return np.matrix([mui - etalambda if mui > etalambda else 0 if - etalambda < mui and mui < etalambda else mui + etalambda for mui in mu.A1.tolist()])

    def updatePG(self):
        self.w = self.prox_eta(self.w - self.eta * self.__phiPrime(self.w))

    def updateAPG(self):
        prevW = self.w
        s = self.s
        self.s = 0.5 * (1 + np.sqrt(1 + 4 * self.s ** 2))
        v = self.w + (s - 1) * (self.w - self.prevW) / self.s
        self.w = self.prox_eta(v - self.eta * self.__phiPrime(v))
        self.prevW = prevW

    def solve(self):
        for _ in range(self.tmax):
            self.update()
            self.history.append(self.w)
