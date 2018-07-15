#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys

from GaussianKernel import GaussianKernel
from LogisticRegression import LogisticRegression

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: %s <filename> <alpha> <dim>\n" % sys.argv[0])
        quit(1)

    filename, alpha, dim = sys.argv[1:]
    alpha = float(alpha)
    dim = int(dim)
    with open(filename, "r") as f:
        next(f)
        data = [([x, y], label) for (x, y, label) in (s.split() for s in f)]
        data = list(zip(*data))

    X = np.matrix([[float(x), float(y)] for (x, y) in data[0]])
    y = np.matrix([int(label) for label in data[1]]).T
    n = X.shape[0]

    data = list(zip(*data))
    z_positive = list(zip(*[t[0] for t in data if t[1] == "1"]))
    z_negative = list(zip(*[t[0] for t in data if t[1] == "-1"]))
    x_positive = [float(s) for s in z_positive[0]]
    y_positive = [float(s) for s in z_positive[1]]
    x_negative = [float(s) for s in z_negative[0]]
    y_negative = [float(s) for s in z_negative[1]]

    gk = GaussianKernel(X, alpha=alpha, dim=dim)
    PhiX = gk.phiX(X)
    model = LogisticRegression(y, PhiX, standardization=False, tmax=-1)
    model.solve()

    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(gk.phiX(np.matrix(list(zip(sum(xx.tolist(), []), sum(yy.tolist(), []))))))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)
    plt.scatter(X[:, 0].A1, X[:, 1].A1, c=y.A1, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig("%d-%d-%d.pdf" % (n, int(alpha), dim))

