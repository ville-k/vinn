[![Build Status](https://travis-ci.org/ville-k/vinn.svg?branch=develop)](https://travis-ci.org/ville-k/vinn)
[![Coverage Status](https://coveralls.io/repos/ville-k/vinn/badge.svg?branch=develop)](https://coveralls.io/r/ville-k/vinn?branch=develop)

# ViNN - A modular OpenCL accelerated library for Deep Learning

ViNN is a cross platform MIT-licensed C++ library for training and evaluating artificial neural networks using OpenCL. ViNN parallelizes computationally intensive linear algebra routines using OpenCL, which provides significant performance gains over a single threaded CPU based implementation. Python bindings with NumPy integration ease developing new models and enable interoperability with existing tools and libraries.

The design goals for ViNN are
* correctness - use gradient checking, static analysis and strive for 100% unit test coverage
* modularity  - for easy integration, extensibility and testing
* performance - optimize performance critical parts using OpenCL


## Features

### Linear Algebra

ViNN ships with a CPU/C++ based and an OpenCL based linear algebra backend
that supports most common matrix operations needed for neural networks
and machine learning applications.

### Activation Functions

* Linear
* Sigmoid
* Softmax
* Hyperbolic Tangent


### Cost Functions

* Squared error
* Cross entropy


### Training

* Batch gradient descent with early stopping
* Stochastic/Minibatch gradient descent with early stopping
* L2 regularization


### Result Measurements

* Confusion table for binary classification
* Multiclass performance measures (average, micro, macro)


### Input/Output Formats

* CSV
* [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) format


## Developing ViNN

### Depencies

Integrating with a project:
* c++ compiler with c++11 support (clang or gcc >= 4.8)
* OpenCL 1.1 driver and development headers
* Swig 3

Additional dependencies when building from source:
* Ruby
* Python 3, NumPy

Additional dependencies for committers:
* clang-format
* lcov

#### Mac OS X (10.10)

    brew install lcov
    brew install clang-format
    brew install swig


#### On Ubuntu (15.04)

    sudo apt-get install build-essential ocl-icd-libopencl1 opencl-headers ruby

#### On Debian Wheezy or Raspbian

    sudo apt-get install gcc-4.8
    sudo apt-get install g++-4.8
    sudo apt-get install libgl1-mesa-dev

### Building from Source

1. Create build directory & configure Debug build:

    mkdir build
    cd build
    # On Ubuntu 15 and Mac OS X:
    cmake -DCMAKE_BUILD_TYPE=Debug ..

    # On Debian Wheezy or Raspbian
    CC=gcc-4.8 CXX=g++-4.8 cmake -DCMAKE_BUILD_TYPE=Debug ..

2. Build library and unit tests

    make

3. Run all tests:

    make test # OR
    ctest

4. Run unit tests & generate coverage report

    make coverage

5. Format source code after making changes:

    make format

6. Build package

    cpack --config CPackConfig.cmake

7. Build Python bindings

    # Source distribution
    cd build/bindings/python
    python setup.py sdist
    pip install --global-option=build_ext --global-option="--swig-opts=-I/Users/ville/projects/vinn/src -c++"  dist/vinnpy-0.2.0.tar.gz

    # Binary wheel distribution
    cd build/bindings/python
    python setup.py build_ext --swig-opts="-I/Users/ville/projects/vinn/src -c++" bdist_wheel
