#!/bin/sh

# Download and unzip MNIST datasets with 780 features scaled to [0, 1]
curl -O http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
curl -O http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
bunzip2 mnist.scale.bz2
bunzip2 mnist.scale.t.bz2
