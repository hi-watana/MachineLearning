#!/bin/sh
for filename in dataset1-{50,100,200}.dat
do
    for alpha in 1 10 100
    do
        for dim in 10 49
        do
            ./gaussianKernelMain.py ${filename} ${alpha} ${dim}
        done
    done
done
