#!/bin/sh
./gaussianKernelMain.py dataset1-100.dat 10 40

./gaussianKernelMain.py dataset1-50.dat 10 40
./gaussianKernelMain.py dataset1-200.dat 10 40

./gaussianKernelMain.py dataset1-100.dat 10 10
./gaussianKernelMain.py dataset1-100.dat 10 70

./gaussianKernelMain.py dataset1-100.dat 1 40
./gaussianKernelMain.py dataset1-100.dat 100 40
