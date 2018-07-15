# MachineLearning

<a name="dataset"></a>
## Dataset
I implemented a script for generating toy datasets which will be used later.

```
$ cd src
$ ./makeData.py
```

After this operation, the following files will be generated:
* `dataset1-50.py` (used in Gaussian Kernel section)
* `dataset1-100.py` (used in Gaussian Kernel section)
* `dataset1-200.py` (used in Gaussian Kernel section)
* `dataset2.py` (used in Logistic Regression section)

## Logistic Regression
I implemented logistic regression. I chose two optimization methods, steepest gradient descent method and newton method.
Source codes are as follows:
* `src/LogisticRegression.py` : Implementation of logistic regression.
* `src/logisticRegressionMain.py` : File for running an experiment on logistic regression.

```
$ cd src
$ ./logisticRegressionMain.py
```

## Lasso
I implemented lasso (Least Absolute Shrinkage and Selection Operator). I chose two optimization methods, proximal gradient (PG) method and accelerated proximal gradient (APG) method.
Source codes are as follows:
* `src/Lasso.py` : Implementation of lasso.
* `src/lassoMain.py` : File for running an experiment on lasso.

```
$ cd src
$ ./lassoMain.py
```

## Gaussian Kernel
I implemented Gaussian kernel. I conducted non-linear classification of points in a [toy dataset](#dataset).
Source codes are as follows:
* `src/GaussianKernel.py` : Implementation of Gaussian kernel
* `src/gaussianKernelMain.py` : File for running an experiment on Gaussian kernel.
* `src/gaussian.sh` : Shellscript used to execute gaussianKernelMain.py repeatedly.

```
$ cd src
$ ./gaussian.sh
```
