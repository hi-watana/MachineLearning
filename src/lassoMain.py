#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from Lasso import Lasso
from Lasso import APG
from Lasso import PG

if __name__ == "__main__":
    for lam, wAns in [(2, np.matrix([0.818, 1.091])), (4, np.matrix([0.636, 0.182])), (6, np.matrix([0.333, 0.]))]:
        modelPG = Lasso(np.matrix([1, 2]), np.matrix([[3, 0.5], [0.5, 1]]), lam, method=PG, tmax=50)
        modelPG.solve()
        modelAPG = Lasso(np.matrix([1, 2]), np.matrix([[3, 0.5], [0.5, 1]]), lam, method=APG, tmax=50)
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
