# ViNN - A modular OpenCL accelerated library for Deep Learning

ViNN is a cross platform MIT-licensed C++ library for training and evaluating artificial neural networks using OpenCL. ViNN parallelizes computationally intensive linear algebra routines using OpenCL, which provides significant performance gains over a single threaded CPU based implementation. 

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


### Result Measurements

* Confusion table for binary classification
* Multiclass performance measures (average, micro, macro)


### Input/Output Formats

* CSV
* [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) format


## Developing ViNN

### Depencies

Integrating with a project:
* c++ compiler with c++11 support (clang or gcc)
* OpenCL 1.2 driver and development headers

Additional dependencies when building from source:
* ruby

Additional dependencies for committers:
* clang-format
* lcov

#### Mac OS X (10.10)

    brew install lcov
    brew install clang-format

#### On Ubuntu (15.04)

    sudo apt-get install build-essential ocl-icd-libopencl1 opencl-headers ruby


### Building from Source

1. Create build directory & configure Debug build:

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Debug ..

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


