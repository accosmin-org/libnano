#include <utest/utest.h>
#include "core/numeric.h"
#include "tensor/tensor.h"
#include <vector>

using namespace nano;

UTEST_BEGIN_MODULE(test_tensor_tensor)

UTEST_CASE(tensor3d)
{
        using tensor3d_t = nano::tensor_mem_t<int, 3>;

        const auto dims = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor3d_t tensor;
        tensor.resize(dims, rows, cols);

        tensor.zero();
        UTEST_CHECK_EQUAL(tensor.vector().minCoeff(), 0);
        UTEST_CHECK_EQUAL(tensor.vector().maxCoeff(), 0);

        UTEST_CHECK_EQUAL(tensor.size<0>(), dims);
        UTEST_CHECK_EQUAL(tensor.size<1>(), rows);
        UTEST_CHECK_EQUAL(tensor.size<2>(), cols);
        UTEST_CHECK_EQUAL(tensor.rows(), rows);
        UTEST_CHECK_EQUAL(tensor.cols(), cols);
        UTEST_CHECK_EQUAL(tensor.size(), dims * rows * cols);

        UTEST_CHECK_EQUAL(tensor.vector().size(), dims * rows * cols);
        UTEST_CHECK_EQUAL(tensor.vector(dims / 2).size(), rows * cols);
        UTEST_CHECK_EQUAL(tensor.vector(dims / 2, rows / 2).size(), cols);

        UTEST_CHECK_EQUAL(tensor.matrix(dims - 1).rows(), tensor.rows());
        UTEST_CHECK_EQUAL(tensor.matrix(dims - 1).cols(), tensor.cols());

        tensor(0, 0, 1) = -3;
        tensor(2, 2, 0) = -7;
        UTEST_CHECK_EQUAL(tensor(0, 0, 1), -3);
        UTEST_CHECK_EQUAL(tensor(2, 2, 0), -7);

        tensor.constant(42);
        UTEST_CHECK_EQUAL(tensor.vector().minCoeff(), 42);
        UTEST_CHECK_EQUAL(tensor.vector().maxCoeff(), 42);

        tensor.constant(42);
        tensor.vector(3, 0).setConstant(7);
        UTEST_CHECK_EQUAL(tensor.vector().minCoeff(), 7);
        UTEST_CHECK_EQUAL(tensor.vector().maxCoeff(), 42);
        UTEST_CHECK_EQUAL(tensor.vector().sum(), 42 * dims * rows * cols - (42 - 7) * cols);

        tensor.matrix(3).setConstant(13);
        UTEST_CHECK_EQUAL(tensor.matrix(3).minCoeff(), 13);
        UTEST_CHECK_EQUAL(tensor.matrix(3).maxCoeff(), 13);
}

UTEST_CASE(tensor3d_map)
{
        using tensor3d_t = nano::tensor_mem_t<int, 3>;

        const auto dims = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor3d_t tensor;
        tensor.resize(dims + 1, rows - 3, cols + 2);

        std::vector<int> v;
        v.reserve(dims * rows * cols);
        for (int i = 0; i < dims * rows * cols; ++ i)
        {
                v.push_back(-35 + i);
        }

        const auto tmap = ::nano::map_tensor(v.data(), dims, rows, cols);
        UTEST_CHECK_EQUAL(tmap.size<0>(), dims);
        UTEST_CHECK_EQUAL(tmap.size<1>(), rows);
        UTEST_CHECK_EQUAL(tmap.size<2>(), cols);
        UTEST_CHECK_EQUAL(tmap.rows(), rows);
        UTEST_CHECK_EQUAL(tmap.cols(), cols);
        UTEST_CHECK_EQUAL(tmap.size(), dims * rows * cols);

        for (int d = 0, i = 0; d < dims; ++ d)
        {
                for (int r = 0; r < rows; ++ r)
                {
                        for (int c = 0; c < cols; ++ c, ++ i)
                        {
                                UTEST_CHECK_EQUAL(tmap(d, r, c), -35 + i);
                        }
                }
        }

        for (int i = 0; i < tmap.size(); ++ i)
        {
                UTEST_CHECK_EQUAL(tmap(i), -35 + i);
        }

        tensor = tmap;
        UTEST_CHECK_EQUAL(tensor.size<0>(), dims);
        UTEST_CHECK_EQUAL(tensor.size<1>(), rows);
        UTEST_CHECK_EQUAL(tensor.size<2>(), cols);
        UTEST_CHECK_EQUAL(tensor.rows(), rows);
        UTEST_CHECK_EQUAL(tensor.cols(), cols);

        for (int d = 0, i = 0; d < dims; ++ d)
        {
                for (int r = 0; r < rows; ++ r)
                {
                        for (int c = 0; c < cols; ++ c, ++ i)
                        {
                                UTEST_CHECK_EQUAL(tensor(d, r, c), -35 + i);
                        }
                }
        }

        for (int i = 0; i < tensor.size(); ++ i)
        {
                UTEST_CHECK_EQUAL(tensor(i), -35 + i);
        }
}

UTEST_CASE(tensor4d)
{
        using tensor4d_t = nano::tensor_mem_t<int, 4>;

        const auto dim1 = 2;
        const auto dim2 = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor4d_t tensor;
        tensor.resize(dim1, dim2, rows, cols);

        tensor.zero();
        UTEST_CHECK_EQUAL(tensor.vector().minCoeff(), 0);
        UTEST_CHECK_EQUAL(tensor.vector().maxCoeff(), 0);

        UTEST_CHECK_EQUAL(tensor.size<0>(), dim1);
        UTEST_CHECK_EQUAL(tensor.size<1>(), dim2);
        UTEST_CHECK_EQUAL(tensor.size<2>(), rows);
        UTEST_CHECK_EQUAL(tensor.size<3>(), cols);
        UTEST_CHECK_EQUAL(tensor.rows(), rows);
        UTEST_CHECK_EQUAL(tensor.cols(), cols);
        UTEST_CHECK_EQUAL(tensor.size(), dim1 * dim2 * rows * cols);

        UTEST_CHECK_EQUAL(tensor.vector().size(), dim1 * dim2 * rows * cols);
        UTEST_CHECK_EQUAL(tensor.vector(dim1 / 2).size(), dim2 * rows * cols);
        UTEST_CHECK_EQUAL(tensor.vector(dim1 / 2, dim2 / 2).size(), rows * cols);
        UTEST_CHECK_EQUAL(tensor.vector(dim1 / 2, dim2 / 2, rows / 2).size(), cols);

        UTEST_CHECK_EQUAL(tensor.matrix(dim1 - 1, dim2 - 1).rows(), tensor.rows());
        UTEST_CHECK_EQUAL(tensor.matrix(dim1 - 1, dim2 - 1).cols(), tensor.cols());

        tensor(0, 4, 0, 1) = -3;
        tensor(1, 2, 2, 0) = -7;
        UTEST_CHECK_EQUAL(tensor(0, 4, 0, 1), -3);
        UTEST_CHECK_EQUAL(tensor(1, 2, 2, 0), -7);

        tensor.constant(42);
        UTEST_CHECK_EQUAL(tensor.vector().minCoeff(), 42);
        UTEST_CHECK_EQUAL(tensor.vector().maxCoeff(), 42);

        tensor.vector(0, 3).setConstant(7);
        UTEST_CHECK_EQUAL(tensor.vector().minCoeff(), 7);
        UTEST_CHECK_EQUAL(tensor.vector().maxCoeff(), 42);
        UTEST_CHECK_EQUAL(tensor.vector().sum(), 42 * dim1 * dim2 * rows * cols - (42 - 7) * rows * cols);

        tensor.matrix(0, 3).setConstant(13);
        UTEST_CHECK_EQUAL(tensor.matrix(0, 3).minCoeff(), 13);
        UTEST_CHECK_EQUAL(tensor.matrix(0, 3).maxCoeff(), 13);
}

UTEST_CASE(tensor4d_map)
{
        using tensor4d_t = nano::tensor_mem_t<int, 4>;

        const auto dim1 = 3;
        const auto dim2 = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor4d_t tensor;
        tensor.resize(dim1 + 2, dim2 + 1, rows - 3, cols + 2);

        std::vector<int> v;
        v.reserve(dim1 * dim2 * rows * cols);
        for (int i = 0; i < dim1 * dim2 * rows * cols; ++ i)
        {
                v.push_back(-35 + i);
        }

        const auto tmap = ::nano::map_tensor(v.data(), dim1, dim2, rows, cols);
        UTEST_CHECK_EQUAL(tmap.size<0>(), dim1);
        UTEST_CHECK_EQUAL(tmap.size<1>(), dim2);
        UTEST_CHECK_EQUAL(tmap.size<2>(), rows);
        UTEST_CHECK_EQUAL(tmap.size<3>(), cols);
        UTEST_CHECK_EQUAL(tmap.rows(), rows);
        UTEST_CHECK_EQUAL(tmap.cols(), cols);
        UTEST_CHECK_EQUAL(tmap.size(), dim1 * dim2 * rows * cols);

        for (int d1 = 0, i = 0; d1 < dim1; ++ d1)
        {
                for (int d2 = 0; d2 < dim2; ++ d2)
                {
                        for (int r = 0; r < rows; ++ r)
                        {
                                for (int c = 0; c < cols; ++ c, ++ i)
                                {
                                        UTEST_CHECK_EQUAL(tmap(d1, d2, r, c), -35 + i);
                                }
                        }
                }
        }

        for (int i = 0; i < tmap.size(); ++ i)
        {
                UTEST_CHECK_EQUAL(tmap(i), -35 + i);
        }

        tensor = tmap;
        UTEST_CHECK_EQUAL(tensor.size<0>(), dim1);
        UTEST_CHECK_EQUAL(tensor.size<1>(), dim2);
        UTEST_CHECK_EQUAL(tensor.size<2>(), rows);
        UTEST_CHECK_EQUAL(tensor.size<3>(), cols);
        UTEST_CHECK_EQUAL(tensor.rows(), rows);
        UTEST_CHECK_EQUAL(tensor.cols(), cols);

        for (int d1 = 0, i = 0; d1 < dim1; ++ d1)
        {
                for (int d2 = 0; d2 < dim2; ++ d2)
                {
                        for (int r = 0; r < rows; ++ r)
                        {
                                for (int c = 0; c < cols; ++ c, ++ i)
                                {
                                        UTEST_CHECK_EQUAL(tensor(d1, d2, r, c), -35 + i);
                                }
                        }
                }
        }

        for (int i = 0; i < tensor.size(); ++ i)
        {
                UTEST_CHECK_EQUAL(tensor(i), -35 + i);
        }
}

UTEST_CASE(tensor3d_fill)
{
        using tensor3d_t = nano::tensor_mem_t<double, 3>;

        const auto dims = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor3d_t tensor;
        tensor.resize(dims, rows, cols);

        tensor.zero();
        UTEST_CHECK_EQUAL(tensor.vector().minCoeff(), 0);
        UTEST_CHECK_EQUAL(tensor.vector().maxCoeff(), 0);

        tensor.constant(-4);
        UTEST_CHECK_EQUAL(tensor.vector().minCoeff(), -4);
        UTEST_CHECK_EQUAL(tensor.vector().maxCoeff(), -4);

        tensor.random(-3, +5);
        UTEST_CHECK_GREATER(tensor.vector().minCoeff(), -3);
        UTEST_CHECK_LESS(tensor.vector().maxCoeff(), +5);

        tensor.random(+5, +11);
        UTEST_CHECK_GREATER(tensor.vector().minCoeff(), +5);
        UTEST_CHECK_LESS(tensor.vector().maxCoeff(), +11);
}

UTEST_CASE(tensor4d_reshape)
{
        using tensor4d_t = nano::tensor_mem_t<int, 4>;

        tensor4d_t tensor(5, 6, 7, 8);

        auto reshape4d = tensor.reshape(5, 3, 28, 4);
        UTEST_CHECK_EQUAL(reshape4d.data(), tensor.data());
        UTEST_CHECK_EQUAL(reshape4d.size(), tensor.size());
        UTEST_CHECK_EQUAL(reshape4d.size<0>(), 5);
        UTEST_CHECK_EQUAL(reshape4d.size<1>(), 3);
        UTEST_CHECK_EQUAL(reshape4d.size<2>(), 28);
        UTEST_CHECK_EQUAL(reshape4d.size<3>(), 4);

        auto reshape3d = tensor.reshape(30, 14, 4);
        UTEST_CHECK_EQUAL(reshape3d.data(), tensor.data());
        UTEST_CHECK_EQUAL(reshape3d.size(), tensor.size());
        UTEST_CHECK_EQUAL(reshape3d.size<0>(), 30);
        UTEST_CHECK_EQUAL(reshape3d.size<1>(), 14);
        UTEST_CHECK_EQUAL(reshape3d.size<2>(), 4);

        auto reshape2d = tensor.reshape(30, 56);
        UTEST_CHECK_EQUAL(reshape2d.data(), tensor.data());
        UTEST_CHECK_EQUAL(reshape2d.size(), tensor.size());
        UTEST_CHECK_EQUAL(reshape2d.size<0>(), 30);
        UTEST_CHECK_EQUAL(reshape2d.size<1>(), 56);

        auto reshape1d = tensor.reshape(1680);
        UTEST_CHECK_EQUAL(reshape1d.data(), tensor.data());
        UTEST_CHECK_EQUAL(reshape1d.size(), tensor.size());
        UTEST_CHECK_EQUAL(reshape1d.size<0>(), 1680);
}

UTEST_CASE(tensor4d_subtensor)
{
        using tensor4d_t = nano::tensor_mem_t<int, 4>;

        const auto dim1 = 2;
        const auto dim2 = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor4d_t tensor;
        tensor.resize(dim1, dim2, rows, cols);

        tensor.constant(42);
        UTEST_CHECK_EQUAL(tensor.vector().minCoeff(), 42);
        UTEST_CHECK_EQUAL(tensor.vector().maxCoeff(), 42);

        tensor.constant(42);
        tensor.tensor(1, 2).setConstant(7);
        UTEST_CHECK_EQUAL(tensor.tensor(1, 2).dims(), nano::make_dims(rows, cols));
        UTEST_CHECK_EQUAL(tensor.array(1, 2).minCoeff(), 7);
        UTEST_CHECK_EQUAL(tensor.array(1, 2).maxCoeff(), 7);
        UTEST_CHECK_EQUAL(tensor.array(1, 2).sum(), 7 * rows * cols);
        UTEST_CHECK_EQUAL(tensor.vector().sum(), 42 * dim1 * dim2 * rows * cols - (42 - 7) * rows * cols);

        tensor.constant(42);
        tensor.tensor(1).setConstant(7);
        UTEST_CHECK_EQUAL(tensor.tensor(1).dims(), nano::make_dims(dim2, rows, cols));
        UTEST_CHECK_EQUAL(tensor.array(1).minCoeff(), 7);
        UTEST_CHECK_EQUAL(tensor.array(1).maxCoeff(), 7);
        UTEST_CHECK_EQUAL(tensor.array(1).sum(), 7 * dim2 * rows * cols);
        UTEST_CHECK_EQUAL(tensor.vector().sum(), 42 * dim1 * dim2 * rows * cols - (42 - 7) * dim2 * rows * cols);
}

UTEST_CASE(tensor4d_subtensor_copying)
{
        using tensor4d_t = nano::tensor_mem_t<int, 4>;

        tensor4d_t tensor1(2, 7, 3, 4);
        tensor4d_t tensor2(2, 7, 3, 4);

        tensor1.random();
        tensor2.random();

        tensor1.tensor(0) = tensor2.tensor(0);
        tensor1.tensor(1) = tensor2.tensor(1);

        UTEST_CHECK_EIGEN_CLOSE(tensor1.vector(), tensor2.vector(), 1);
}

UTEST_END_MODULE()
