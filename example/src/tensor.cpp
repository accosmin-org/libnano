#include <iostream>
#include <nano/tensor/stream.h>
#include <sstream>

int main(const int, char*[])
{
    // initialize a random 1D tensor of size 48
    auto tensor = nano::tensor_mem_t<int, 1>{48};
    tensor.random(-10, +10);

    // dimensions
    std::cout << "tensor.rank():\n" << tensor.rank() << "\n\n";
    std::cout << "tensor.dims():\n" << tensor.dims() << "\n\n";
    std::cout << "tensor.size():\n" << tensor.size() << "\n\n";
    std::cout << "tensor.size<0>():\n" << tensor.size<0>() << "\n\n";

    // tensors can be mapped to contiguous Eigen vectors
    // NB: no copying is performed!
    std::cout << "tensor.vector().transpose():\n" << tensor.vector().transpose() << "\n\n";

    // tensors can be mapped to contiguous Eigen arrays as well
    // NB: no copying is performed!
    std::cout << "tensor.array().transpose():\n" << tensor.array().transpose() << "\n\n";

    // and thus all Eigen utilities are available as well
    std::cout << "tensor.vector().minCoeff():\n" << tensor.vector().minCoeff() << "\n\n";
    std::cout << "tensor.array().maxCoeff():\n" << tensor.array().maxCoeff() << "\n\n";

    // basic statistics and reductions can be computed as well using the tensor interface directly
    std::cout << "tensor.min():\n" << tensor.min() << "\n\n";
    std::cout << "tensor.max():\n" << tensor.max() << "\n\n";
    std::cout << "tensor.sum():\n" << tensor.sum() << "\n\n";
    std::cout << "tensor.mean():\n" << tensor.mean() << "\n\n";
    std::cout << "tensor.stdev():\n" << tensor.stdev() << "\n\n";

    // tensors can change shape (if the total number of elements is the same)
    // and 2D tensors can be mapped to row-major contiguous Eigen matrices
    // NB: no copying is performed!
    std::cout << "tensor.reshape(6, 8).matrix():\n" << tensor.reshape(6, 8).matrix() << "\n\n";

    // multi-dimensional tensors can be mapped to contiguous Eigen matrices
    // with the last two dimensions being interpreted as the rows and the columns
    // NB: no copying is performed!
    std::cout << "tensor.reshape(3, 4, 4).matrix(0):\n" << tensor.reshape(3, 4, 4).matrix(0) << "\n\n";
    std::cout << "tensor.reshape(3, 4, 4).matrix(1):\n" << tensor.reshape(3, 4, 4).matrix(1) << "\n\n";
    std::cout << "tensor.reshape(3, 4, 4).matrix(2):\n" << tensor.reshape(3, 4, 4).matrix(2) << "\n\n";
    std::cout << "tensor.reshape(3, 2, 2, 4).matrix(0, 1):\n" << tensor.reshape(3, 2, 2, 4).matrix(0, 1) << "\n\n";

    // multi-dimensional tensors can be mapped to lower-ranked contiguous tensors
    // NB: no copying is performed!
    std::cout << "tensor.reshape(3, 4, 4).tensor(1).dims():\n" << tensor.reshape(3, 4, 4).tensor(1).dims() << "\n\n";
    std::cout << "tensor.reshape(3, 4, 4).tensor(1).matrix():\n"
              << tensor.reshape(3, 4, 4).tensor(1).matrix() << "\n\n";
    std::cout << "tensor.reshape(3, 4, 4).tensor(1, 2).dims():\n" << tensor.reshape(3, 4, 4).tensor(1).dims() << "\n\n";
    std::cout << "tensor.reshape(3, 4, 4).tensor(1, 2).vector():\n"
              << tensor.reshape(3, 4, 4).tensor(1).matrix() << "\n\n";

    // tensors can be sliced only along the first dimension to obtain contiguous tensors with the same rank
    // NB: no copying is performed!
    std::cout << "tensor.reshape(6, 2, 4).slice(2, 5).dims():\n"
              << tensor.reshape(6, 2, 4).slice(2, 5).dims() << "\n\n";

    std::cout << "tensor.reshape(6, 2, 4).slice(2, 5).reshape(3, 8).matrix():\n"
              << tensor.reshape(6, 2, 4).slice(2, 5).reshape(3, 8).matrix() << "\n\n";

    // tensor values can be read and written using the appropriate indices
    std::cout << "tensor(7):\n" << tensor(7) << "\n\n";
    std::cout << "tensor(7) = -100:\n\n";
    tensor(7) = -100;
    std::cout << "tensor(7):\n" << tensor(7) << "\n\n";
    std::cout << "tensor.reshape(3, 4, 4)(0, 1, 3):\n" << tensor.reshape(3, 4, 4)(0, 1, 3) << "\n\n";

    // tensors (as well as Eigen vectors and matrices) can be mapped from (contiguous) mutable or constant C-arrays
    const float carray[12] = {-3.0f, -2.5f, -2.0f, -1.5f, -1.0f, -0.5f, +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +2.5f};

    const auto vector  = nano::map_vector(carray, 12);
    const auto matrix  = nano::map_matrix(carray, 3, 4);
    const auto tensor1 = nano::map_tensor(carray, 12);
    const auto tensor2 = nano::map_tensor(carray, 3, 4);
    const auto tensor3 = nano::map_tensor(carray, 3, 2, 2);

    std::cout << "map_vector(carray, 12):\n" << vector.transpose() << "\n\n";
    std::cout << "map_matrix(carray, 3, 4):\n" << matrix << "\n\n";
    std::cout << "map_tensor(carray, 12).dims():\n" << tensor1.dims() << "\n\n";
    std::cout << "map_tensor(carray, 3, 4).dims():\n" << tensor2.dims() << "\n\n";
    std::cout << "map_tensor(carray, 3, 2, 2).dims():\n" << tensor3.dims() << "\n\n";
    std::cout << "map_tensor(carray, 3, 2, 2).data() - carray:\n"
              << static_cast<int>(tensor3.data() - carray) << "\n\n";

    // tensors can be written and read unformatted to and from STL streams
    std::string buffer;
    {
        std::ostringstream ostream;
        nano::write(ostream, tensor);
        buffer = ostream.str();
    }
    decltype(tensor) read_tensor;
    {
        std::istringstream istream(buffer);
        nano::read(istream, read_tensor);
    }
    std::cout << "(tensor.vector() - read_tensor.vector()).transpose():\n"
              << (tensor.vector() - read_tensor.vector()).transpose() << "\n";

    return EXIT_SUCCESS;
}
