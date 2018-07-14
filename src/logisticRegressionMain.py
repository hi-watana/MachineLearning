#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from LogisticRegression import LogisticRegression
from LogisticRegression import GRAD
from LogisticRegression import NEWTON

if __name__ == "__main__":
    with open("dataset2.dat", "r") as f:
        next(f)
        data = [([x, y], label) for (x, y, label) in (s.split() for s in f)]
        data = list(zip(*data))

    X = np.matrix([[float(x), float(y)] for (x, y) in data[0]])
    y = np.matrix([int(label) for label in data[1]]).T

    data = list(zip(*data))
    z_positive = list(zip(*[t[0] for t in data if t[1] == "1"]))
    z_negative = list(zip(*[t[0] for t in data if t[1] == "-1"]))
    x_positive = [float(s) for s in z_positive[0]]
    y_positive = [float(s) for s in z_positive[1]]
    x_negative = [float(s) for s in z_negative[0]]
    y_negative = [float(s) for s in z_negative[1]]

    models = [None] * 2
    methods = [GRAD, NEWTON]
    methodNames = {GRAD: "steepest gradient method", NEWTON: "newton method"}
    for i, method in enumerate(methods):
        models[i] = LogisticRegression(y, X, method, tmax=10, lam=1)
        models[i].solve()

    # figure about classification
    plt.plot(x_positive, y_positive, "o")
    plt.plot(x_negative, y_negative, "o")

    for model, method in zip(models, methods):
        wlist = model.w.tolist()[0]
        points_on_boundary = [(np.matrix([-wlist[1], wlist[0]]) * model.C * t + model.mu).tolist()[0] for t in np.arange(-4.0, 4.0, 0.1)]
        x_points, y_points = zip(*points_on_boundary)
        plt.plot(x_points, y_points, label=methodNames[method])

    plt.legend(loc = "upper left")
    plt.savefig("classification.pdf")

    plt.figure() # clear graph

    # figure about convergence of w
    for model, method in zip(models, methods):
        #w1_list = [w1 / w2 for (w1, w2) in model.history]
        t_list = list(range(len(model.history)))
        jw_list = [model.lossFunction(np.matrix(w)) for w in model.history]
        plt.plot(t_list, jw_list, label=methodNames[method])
        #plt.plot(w1_list, jw_list, ".")
    plt.xlabel("t")
    plt.ylabel("J(w)")
    plt.legend(loc = "upper right")
    plt.savefig("convergent_sequence.pdf", bbox_inches="tight")
