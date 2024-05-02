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
OMP_NUM_THREADS=4 ./a.out nips.tns 4d.meta 0 1
```

Produces the following ouput:

<pre><code>
2482 2862 14036 17
Number of elements inserted = 3101609
Total time for insertion = 0.76703 (s)
Number of elements inserted = 3101609
Total time for insertion = 0.101242 (s)
Total time for preprocessing = 0.65588 (s)
Total time for tensor contraction = 2.12908 (s)
Total number of nonzeros in the output tensor = 28334
</code></pre>


