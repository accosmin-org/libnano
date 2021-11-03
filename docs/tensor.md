# Tensor module


#### Introduction

Eigen matrices and vectors and the builtin tensors constitute the core data structures for all numerical optimization and the machine learning utilities provided by libnano. For example, the gradient of a function to minimize is stored as an Eigen vector, while a fixed-size image classification dataset is stored as a 4D tensor with dimensions number of images, number of color channels, image rows and image columns.

This section describes the tensor module built on top of [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#Documentation) data structures. The following lists the main properties of the tensor module:
* [header-only](../include/nano/tensor/).
* uses Eigen data structures for storage and numerical operations.
* supports tensors of arbitrary rank using all builtin scalar types (e.g. int, float, double).
* data is stored contiguously and all operations return lightweight proxies for full compatibility with C-arrays and STL algorithms. As such only row-major Eigen data structures are supported.

A good starting point for understanding how to use the tensor module is to read the related unit tests (see [test_tensor_index](../test/test_tensor_index.cpp), [test_tensor_tensor](../test/test_tensor_tensor.cpp), [test_tensor_stream](../test/test_tensor_stream.cpp) or [test_tensor_integral](../test/test_tensor_integral.cpp) or the [example](../example/src/tensor.cpp) presented in some detail bellow.

#### Examples

Allocate a 1D tensor of integers of size 48 and then initialize it from a random uniform distribution in the range \[-10, +10\]:
```
#include <iostream>
#include <nano/tensor/tensor.h>

auto tensor = nano::tensor_mem_t<int, 1>{48};
tensor.random(-10, +10);
```

Print the rank, the dimensions and the size of this tensor:
```
std::cout << tensor.rank() << std::endl;    // prints: 1
std::cout << tensor.dims() << std::endl;    // prints: 48
std::cout << tensor.size() << std::endl;    // prints: 48
std::cout << tensor.size<0>() << std::endl; // prints: 48
```


All tensors can be represented as Eigen arrays or vectors and as such all supported Eigen [operations](https://eigen.tuxfamily.org/dox/group__DenseMatrixManipulation__chapter.html) can be performed on tensors as well:
```
std::cout << tensor.vector().transpose() << std::endl;
// prints:
//  -7   7  -8  -7   1  -9  -5   7  -8  -1  10  -8  10  -9   4  10  ...
//  -6  -9   3  -4   3  -4  -8   3   4   3  10   3   9  -7   9   3  ...
//   5   2  -1   9   5  -6  -9 -10  -7  -5   4   9  10   4  -2  10
std::cout << tensor.vector().minCoeff() << std::endl; // prints: -10
std::cout << tensor.vector().maxCoeff() << std::endl; // prints: 10
```

Tensors can be reshaped to tensors of potentially different rank if their total size match. In this case the 1D tensor of dimension 48 can be reshaped to a 2D tensor of dimensions (6, 8):
```
std::cout << tensor.reshape(6, 8).matrix() << std::endl;
// prints:
// -7   7  -8  -7   1  -9  -5   7
// -8  -1  10  -8  10  -9   4  10
// -6  -9   3  -4   3  -4  -8   3
//  4   3  10   3   9  -7   9   3
//  5   2  -1   9   5  -6  -9 -10
// -7  -5   4   9  10   4  -2  10
```

Another useful operation is to slice a given tensor. Please note that slicing is only supported along the first dimension to make sure the returned proxy maps contiguous data. As such the returned proxy tensor has the same rank as the original tensor. The following examples combine reshaping and slicing operations on the original tensor:
```
std::cout << tensor.reshape(6, 2, 4).slice(2, 5).dims() << std::endl; // prints: 3x2x4
std::cout << tensor.reshape(6, 2, 4).slice(2, 5).reshape(3, 8).matrix() << std::endl;
// prints:
// -6  -9   3  -4   3  -4  -8   3
//  4   3  10   3   9  -7   9   3
//  5   2  -1   9   5  -6  -9 -10
```

Individual elements are accessable by mutable or constant reference using the following intuitive notation similar to the one used by Eigen data structures:
```
std::cout << tensor(7) << std::endl;                        // prints: 7
tensor(7) = -100; std::cout << tensor(7) << std::endl;      // prints: -100
std::cout << tensor.reshape(3, 4, 4)(0, 1, 3) << std::endl; // prints: -100
```

Another useful operation is to map contiguous mutable or constant C-arrays to Eigen vectors and matrices or to tensors. Please note that these utilities are also used for various operations on tensors, like slicing or reshaping.
```
const float carray[12] = {
    -3.0f, -2.5f, -2.0f, -1.5f, -1.0f, -0.5f,
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +2.5f};

const auto vector = nano::map_vector(carray, 12);
const auto matrix = nano::map_matrix(carray, 3, 4);
const auto tensor1 = nano::map_tensor(carray, 12);
const auto tensor2 = nano::map_tensor(carray, 3, 4);
const auto tensor3 = nano::map_tensor(carray, 3, 2, 2);

std::cout << vector.transpose() << std::endl;
// prints:  -3 -2.5   -2 -1.5   -1 -0.5    0  0.5    1  1.5    2  2.5
std::cout << matrix << std::endl;
// prints:
// -3 -2.5   -2 -1.5
// -1 -0.5    0  0.5
//  1  1.5    2  2.5
std::cout << tensor1.dims() << std::endl; // prints: 12
std::cout << tensor2.dims() << std::endl; // prints: 3x4
std::cout << tensor3.dims() << std::endl; // prints: 3x2x2
std::cout << static_cast<int>(tensor3.data() - carray) << std::endl;
// prints: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

#### Other related utilities

* serialization (see [stream](../include/nano/tensor/stream.h))
* summed-area tables of tensors of arbitrary rank (see [integral](../include/nano/tensor/integral.h))
* command line utility to benchmark [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) operations using Eigen vectors and matrices (see [app/bench_eigen](../app/bench_eigen.cpp))
