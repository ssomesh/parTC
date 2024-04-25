# Efficient Parallel Sparse Tensor Contraction

This repository contains the source code accompanying the manuscript 

**Somesh Singh and Bora Uçar**. *Efficient Parallel Sparse Tensor Contraction*.

## Building and Usage

### Prerequisites
* g++ (gcc) version 13.2.0 or higher, with support for *OpenMP*
* [Boost](https://www.boost.org/) C++ libraries version 1.67.0 or higher

### Building

The codes are written in C++.

* To build the code run `make`

The executable will be generated within the top-level directory of the repository

### Usage

* For running experiments on real-life tensors from [FROSTT](http://frostt.io/tensors/)
```
OMP_NUM_THREADS=1 ./a.out /warehouse/bucar/nips.tns /warehouse/ssingh01/tc_results/dimN_4_R_2_4.meta 0 1
```
