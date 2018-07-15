#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from Lasso import Lasso
from Lasso import APG
from Lasso import PG

if __name__ == "__main__":
    tmax = 80
    for lam, wAns in [(2, np.matrix([0.81818, 1.09091])), (4, np.matrix([0.63636, 0.18182])), (6, np.matrix([0.33333, 0.]))]:
        modelPG = Lasso(np.matrix([1, 2]), np.matrix([[3, 0.5], [0.5, 1]]), lam, method=PG, tmax=tmax)
        modelPG.solve()
        modelAPG = Lasso(np.matrix([1, 2]), np.matrix([[3, 0.5], [0.5, 1]]), lam, method=APG, tmax=tmax)
        modelAPG.solve()
        yPG = [np.sqrt(dw * dw.T).tolist()[0][0] for dw in (w - wAns for w in modelPG.history)]
        yAPG = [np.sqrt(dw * dw.T).tolist()[0][0] for dw in (w - wAns for w in modelAPG.history)]
        print(modelPG.w)
        print(modelAPG.w)
        x = list(range(len(yPG)))
        plt.plot(x, yPG, label="PG")
        plt.plot(x, yAPG, label="APG")
        plt.xlabel("t")
        plt.ylabel("||w - w^|||_2")
        plt.yscale("log")
        plt.legend(loc = "upper right")
        plt.savefig("lasso-%d.pdf" % lam, bbox_inches="tight")
        plt.figure()
