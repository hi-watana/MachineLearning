#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # dataset 1
    for n in [50, 100, 200]:
        np.random.seed(3)
        x = 3 * (np.random.rand(n, 2) - 0.5)
        radius = (x.T[0] ** 2 + x.T[1] ** 2).tolist()
        y = [((r > 0.7 + 0.1 * rand1) and (r < 2.2 + 0.1 * rand2)) for (r, rand1, rand2) in zip(radius, np.random.randn(n).tolist(), np.random.randn(n).tolist())]
        y = [1 if b else -1 for b in y]
        with open("dataset1-%d.dat" % n, "w") as f:
            f.write("x y label\n")
            for xi, yi in zip(x, y):
                f.write("%f %f %d\n" % (xi[0], xi[1], yi))


    # dataset 2
    n = 40
    np.random.seed(1000)
    omega = np.random.randn(1, 1)
    noise = 0.8 * np.random.randn(n, 1)

    x = np.random.randn(n, 2)
    x0 = x.T[0].reshape(n, 1)
    x1 = x.T[1].reshape(n, 1)
    y = 2 * (omega * x0 + x1 + noise > 0) - 1
    with open("dataset2.dat", "w") as f:
        f.write("x y label\n")
        for xi, yi in zip(x, y):
            f.write("%f %f %d\n" % (xi[0], xi[1], yi[0]))
