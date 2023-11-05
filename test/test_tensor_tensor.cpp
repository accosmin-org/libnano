#include <nano/core/numeric.h>
#include <nano/tensor/tensor.h>
#include <utest/utest.h>
#include <vector>

using namespace nano;

UTEST_BEGIN_MODULE(test_tensor_tensor)

UTEST_CASE(print)
{
    const auto vector   = ::nano::arange(0, 24);
    const auto u8vector = ::nano::make_tensor<uint8_t>(make_dims(24), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                       15, 16, 17, 18, 19, 20, 21, 22, 23);
    const auto i8vector = ::nano::make_tensor<int8_t>(make_dims(24), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                      15, 16, 17, 18, 19, 20, 21, 22, 23);
    {
        const auto* const expected = R"(shape: 24
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23])";

        {
            std::ostringstream stream;
            stream << vector;
            UTEST_CHECK_EQUAL(stream.str(), expected);
        }
        {
            std::ostringstream stream;
            stream << u8vector;
            UTEST_CHECK_EQUAL(stream.str(), expected);
        }
        {
            std::ostringstream stream;
            stream << i8vector;
            UTEST_CHECK_EQUAL(stream.str(), expected);
        }
    }
    {
        const auto* const expected = R"(shape: 4x6
[[0 1 2 3 4 5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]])";

        {
            std::ostringstream stream;
            stream << vector.reshape(4, -1);
            UTEST_CHECK_EQUAL(stream.str(), expected);
        }
        {
            std::ostringstream stream;
            stream << u8vector.reshape(4, -1);
            UTEST_CHECK_EQUAL(stream.str(), expected);
        }
        {
            std::ostringstream stream;
            stream << i8vector.reshape(4, -1);
            UTEST_CHECK_EQUAL(stream.str(), expected);
        }
    }
    {
        const auto* const expected = R"(shape: 4x3x2
[[[0 1]
  [2 3]
  [4 5]]
 [[6 7]
  [8 9]
  [10 11]]
 [[12 13]
  [14 15]
  [16 17]]
 [[18 19]
  [20 21]
  [22 23]]])";

        std::ostringstream stream;
        stream << vector.reshape(4, 3, -1);
        UTEST_CHECK_EQUAL(stream.str(), expected);
    }
    {
        const auto* const expected = R"(shape: 4x3x1x2
[[[[0 1]]
  [[2 3]]
  [[4 5]]]
 [[[6 7]]
  [[8 9]]
  [[10 11]]]
 [[[12 13]]
  [[14 15]]
  [[16 17]]]
 [[[18 19]]
  [[20 21]]
  [[22 23]]]])";

        std::ostringstream stream;
        stream << vector.reshape(4, 3, 1, -1);
        UTEST_CHECK_EQUAL(stream.str(), expected);
    }
}

UTEST_CASE(tensor3d)
{
    using tensor3d_t = nano::tensor_mem_t<int, 3>;

    const auto dims = tensor_size_t{7};
    const auto rows = tensor_size_t{3};
    const auto cols = tensor_size_t{4};

    tensor3d_t tensor;
    tensor.resize(dims, rows, cols);
    tensor.zero();

    UTEST_CHECK_EQUAL(tensor.min(), 0);
    UTEST_CHECK_EQUAL(tensor.max(), 0);
    UTEST_CHECK_EQUAL(tensor.sum(), 0);
    UTEST_CHECK_EQUAL(tensor.mean(), 0);

    UTEST_CHECK_EQUAL(tensor.rank(), 3);
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

    tensor.full(42);
    UTEST_CHECK_EQUAL(tensor.min(), 42);
    UTEST_CHECK_EQUAL(tensor.max(), 42);
    UTEST_CHECK_EQUAL(tensor.sum(), 42 * tensor.size());
    UTEST_CHECK_EQUAL(tensor.mean(), 42);

    tensor.full(42);
    tensor.vector(3, 0).setConstant(7);
    UTEST_CHECK_EQUAL(tensor.min(), 7);
    UTEST_CHECK_EQUAL(tensor.max(), 42);
    UTEST_CHECK_EQUAL(tensor.sum(), cols * 7 + (tensor.size() - cols) * 42);
    UTEST_CHECK_EQUAL(tensor.mean(), (cols * 7 + (tensor.size() - cols) * 42) / tensor.size());
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
    v.reserve(static_cast<size_t>(dims) * static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int i = 0; i < dims * rows * cols; ++i)
    {
        v.push_back(-35 + i);
    }

    auto tmap = ::nano::map_tensor(v.data(), dims, rows, cols);
    UTEST_CHECK_EQUAL(tmap.rank(), 3);
    UTEST_CHECK_EQUAL(tmap.size<0>(), dims);
    UTEST_CHECK_EQUAL(tmap.size<1>(), rows);
    UTEST_CHECK_EQUAL(tmap.size<2>(), cols);
    UTEST_CHECK_EQUAL(tmap.rows(), rows);
    UTEST_CHECK_EQUAL(tmap.cols(), cols);
    UTEST_CHECK_EQUAL(tmap.size(), dims * rows * cols);

    for (tensor_size_t d = 0, i = 0; d < dims; ++d)
    {
        for (tensor_size_t r = 0; r < rows; ++r)
        {
            for (tensor_size_t c = 0; c < cols; ++c, ++i)
            {
                UTEST_CHECK_EQUAL(tmap(d, r, c), -35 + i);
            }
        }
    }

    for (tensor_size_t i = 0; i < tmap.size(); ++i)
    {
        UTEST_CHECK_EQUAL(tmap(i), -35 + i);
    }

    tensor = tmap;
    UTEST_CHECK_EQUAL(tensor.size<0>(), dims);
    UTEST_CHECK_EQUAL(tensor.size<1>(), rows);
    UTEST_CHECK_EQUAL(tensor.size<2>(), cols);
    UTEST_CHECK_EQUAL(tensor.rows(), rows);
    UTEST_CHECK_EQUAL(tensor.cols(), cols);

    for (tensor_size_t d = 0, i = 0; d < dims; ++d)
    {
        for (tensor_size_t r = 0; r < rows; ++r)
        {
            for (tensor_size_t c = 0; c < cols; ++c, ++i)
            {
                UTEST_CHECK_EQUAL(tensor(d, r, c), -35 + i);
            }
        }
    }

    for (tensor_size_t i = 0; i < tensor.size(); ++i)
    {
        UTEST_CHECK_EQUAL(tensor(i), -35 + i);
    }

    tmap.random();

    for (tensor_size_t i = 0; i < tensor.size(); ++i)
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
    UTEST_CHECK_EQUAL(tensor.min(), 0);
    UTEST_CHECK_EQUAL(tensor.max(), 0);

    UTEST_CHECK_EQUAL(tensor.rank(), 4);
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

    tensor.full(42);
    UTEST_CHECK_EQUAL(tensor.min(), 42);
    UTEST_CHECK_EQUAL(tensor.max(), 42);

    tensor.vector(0, 3).setConstant(7);
    UTEST_CHECK_EQUAL(tensor.min(), 7);
    UTEST_CHECK_EQUAL(tensor.max(), 42);
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
    v.reserve(static_cast<size_t>(dim1) * static_cast<size_t>(dim2) * static_cast<size_t>(rows) *
              static_cast<size_t>(cols));
    for (int i = 0; i < dim1 * dim2 * rows * cols; ++i)
    {
        v.push_back(-35 + i);
    }

    const auto tmap = ::nano::map_tensor(v.data(), dim1, dim2, rows, cols);
    UTEST_CHECK_EQUAL(tmap.rank(), 4);
    UTEST_CHECK_EQUAL(tmap.size<0>(), dim1);
    UTEST_CHECK_EQUAL(tmap.size<1>(), dim2);
    UTEST_CHECK_EQUAL(tmap.size<2>(), rows);
    UTEST_CHECK_EQUAL(tmap.size<3>(), cols);
    UTEST_CHECK_EQUAL(tmap.rows(), rows);
    UTEST_CHECK_EQUAL(tmap.cols(), cols);
    UTEST_CHECK_EQUAL(tmap.size(), dim1 * dim2 * rows * cols);

    for (tensor_size_t d1 = 0, i = 0; d1 < dim1; ++d1)
    {
        for (tensor_size_t d2 = 0; d2 < dim2; ++d2)
        {
            for (tensor_size_t r = 0; r < rows; ++r)
            {
                for (tensor_size_t c = 0; c < cols; ++c, ++i)
                {
                    UTEST_CHECK_EQUAL(tmap(d1, d2, r, c), -35 + i);
                }
            }
        }
    }

    for (tensor_size_t i = 0; i < tmap.size(); ++i)
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

    for (tensor_size_t d1 = 0, i = 0; d1 < dim1; ++d1)
    {
        for (tensor_size_t d2 = 0; d2 < dim2; ++d2)
        {
            for (tensor_size_t r = 0; r < rows; ++r)
            {
                for (tensor_size_t c = 0; c < cols; ++c, ++i)
                {
                    UTEST_CHECK_EQUAL(tensor(d1, d2, r, c), -35 + i);
                }
            }
        }
    }

    for (tensor_size_t i = 0; i < tensor.size(); ++i)
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
    UTEST_CHECK_EQUAL(tensor.min(), 0);
    UTEST_CHECK_EQUAL(tensor.max(), 0);

    tensor.full(-4);
    UTEST_CHECK_EQUAL(tensor.min(), -4);
    UTEST_CHECK_EQUAL(tensor.max(), -4);

    tensor.random(-3, +5);
    UTEST_CHECK_GREATER(tensor.min(), -3);
    UTEST_CHECK_LESS(tensor.max(), +5);

    tensor.random(+5, +11);
    UTEST_CHECK_GREATER(tensor.min(), +5);
    UTEST_CHECK_LESS(tensor.max(), +11);
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

    auto reshape3d = tensor.reshape(30, -1, 4);
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

    tensor.full(42);
    UTEST_CHECK_EQUAL(tensor.min(), 42);
    UTEST_CHECK_EQUAL(tensor.max(), 42);

    tensor.full(42);
    tensor.tensor(1, 2).full(7);
    UTEST_CHECK_EQUAL(tensor.tensor(1, 2).dims(), nano::make_dims(rows, cols));
    UTEST_CHECK_EQUAL(tensor.array(1, 2).minCoeff(), 7);
    UTEST_CHECK_EQUAL(tensor.array(1, 2).maxCoeff(), 7);
    UTEST_CHECK_EQUAL(tensor.array(1, 2).sum(), 7 * rows * cols);
    UTEST_CHECK_EQUAL(tensor.vector().sum(), 42 * dim1 * dim2 * rows * cols - (42 - 7) * rows * cols);

    tensor.full(42);
    tensor.tensor(1).full(7);
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

    UTEST_CHECK_EQUAL(tensor1, tensor2);
}

UTEST_CASE(tensor1d_indexing)
{
    using tensor1d_t = nano::tensor_mem_t<int16_t, 1>;

    tensor1d_t tensor(13);
    tensor.random();

    const auto indices   = make_tensor<tensor_size_t>(make_dims(6), 0, 1, 3, 2, 2, 7);
    const auto subtensor = tensor.indexed<int32_t>(indices);

    UTEST_REQUIRE_EQUAL(subtensor.size<0>(), 6);

    UTEST_CHECK_EQUAL(static_cast<int16_t>(subtensor(0)), tensor(0));
    UTEST_CHECK_EQUAL(static_cast<int16_t>(subtensor(1)), tensor(1));
    UTEST_CHECK_EQUAL(static_cast<int16_t>(subtensor(2)), tensor(3));
    UTEST_CHECK_EQUAL(static_cast<int16_t>(subtensor(3)), tensor(2));
    UTEST_CHECK_EQUAL(static_cast<int16_t>(subtensor(4)), tensor(2));
    UTEST_CHECK_EQUAL(static_cast<int16_t>(subtensor(5)), tensor(7));
}

UTEST_CASE(tensor4d_indexing)
{
    using tensor4d_t = nano::tensor_mem_t<int16_t, 4>;

    tensor4d_t tensor(5, 7, 3, 4);
    tensor.random();

    const auto indices   = make_tensor<tensor_size_t>(make_dims(6), 0, 1, 3, 2, 2, 3);
    const auto subtensor = tensor.indexed<int32_t>(indices);

    UTEST_REQUIRE_EQUAL(subtensor.size<0>(), 6);
    UTEST_REQUIRE_EQUAL(subtensor.size<1>(), tensor.size<1>());
    UTEST_REQUIRE_EQUAL(subtensor.size<2>(), tensor.size<2>());
    UTEST_REQUIRE_EQUAL(subtensor.size<3>(), tensor.size<3>());

    UTEST_CHECK_EQUAL(subtensor.vector(0).cast<int16_t>(), tensor.vector(0));
    UTEST_CHECK_EQUAL(subtensor.vector(1).cast<int16_t>(), tensor.vector(1));
    UTEST_CHECK_EQUAL(subtensor.vector(2).cast<int16_t>(), tensor.vector(3));
    UTEST_CHECK_EQUAL(subtensor.vector(3).cast<int16_t>(), tensor.vector(2));
    UTEST_CHECK_EQUAL(subtensor.vector(4).cast<int16_t>(), tensor.vector(2));
    UTEST_CHECK_EQUAL(subtensor.vector(5).cast<int16_t>(), tensor.vector(3));
}

UTEST_CASE(tensor4d_slice)
{
    using tensor4d_t = nano::tensor_mem_t<int16_t, 4>;

    tensor4d_t tensor(5, 7, 3, 4);
    tensor.random();

    const auto slice1 = tensor.slice(0, 2);
    const auto slice2 = tensor.tensor(2, 3).slice(make_range(1, 2));

    const auto dims1 = nano::make_dims(2, 7, 3, 4);
    const auto dims2 = nano::make_dims(1, 4);

    UTEST_REQUIRE_EQUAL(slice1.dims(), dims1);
    UTEST_REQUIRE_EQUAL(slice2.dims(), dims2);

    UTEST_CHECK_EQUAL(tensor.vector(0), slice1.vector(0));
    UTEST_CHECK_EQUAL(tensor.vector(1), slice1.vector(1));
    UTEST_CHECK_EQUAL(tensor.vector(2, 3, 1), slice2.vector());
}

UTEST_CASE(tensor4d_lin_spaced)
{
    using tensor4d_t = nano::tensor_mem_t<int16_t, 4>;

    tensor4d_t tensor(1, 2, 3, 4);
    tensor.lin_spaced(1, 24);

    for (tensor_size_t i = 0; i < tensor.size(); ++i)
    {
        UTEST_CHECK_EQUAL(tensor(i), static_cast<int16_t>(i + 1));
    }

    const auto indices = ::nano::arange(1, 5);
    UTEST_REQUIRE_EQUAL(indices.size(), 4);
    UTEST_CHECK_EQUAL(indices(0), 1);
    UTEST_CHECK_EQUAL(indices(1), 2);
    UTEST_CHECK_EQUAL(indices(2), 3);
    UTEST_CHECK_EQUAL(indices(3), 4);
}

UTEST_CASE(tensor4d_begin_end)
{
    using tensor4d_t = nano::tensor_mem_t<int16_t, 4>;

    tensor4d_t tensor(1, 2, 3, 4);

    int16_t index = 0;
    for (auto& value : tensor)
    {
        value = index++; // cppcheck-suppress useStlAlgorithm
    }

    for (tensor_size_t i = 0; i < tensor.size(); ++i)
    {
        UTEST_CHECK_EQUAL(tensor(i), static_cast<int16_t>(i));
    }
}

UTEST_CASE(tensor3d_from_array)
{
    const auto tensor = make_tensor<int16_t>(make_dims(3, 2, 1), 0, 1, 10, 11, 20, 21);

    UTEST_CHECK_EQUAL(tensor.size<0>(), 3);
    UTEST_CHECK_EQUAL(tensor.size<1>(), 2);
    UTEST_CHECK_EQUAL(tensor.size<2>(), 1);

    UTEST_CHECK_EQUAL(tensor(0), 0);
    UTEST_CHECK_EQUAL(tensor(1), 1);
    UTEST_CHECK_EQUAL(tensor(2), 10);
    UTEST_CHECK_EQUAL(tensor(3), 11);
    UTEST_CHECK_EQUAL(tensor(4), 20);
    UTEST_CHECK_EQUAL(tensor(5), 21);
}

UTEST_CASE(tensor3d_minmax)
{
    using tensor4d_t = nano::tensor_mem_t<int16_t, 4>;

    tensor4d_t tensor(1, 2, 3, 4);
    tensor.lin_spaced(1, 24);

    UTEST_CHECK_EQUAL(tensor.min(), 1);
    UTEST_CHECK_EQUAL(tensor.max(), 24);

    UTEST_CHECK_EQUAL(tensor.tensor(0, 0).min(), 1);
    UTEST_CHECK_EQUAL(tensor.tensor(0, 0).max(), 12);

    UTEST_CHECK_EQUAL(tensor.tensor(0, 1).min(), 13);
    UTEST_CHECK_EQUAL(tensor.tensor(0, 1).max(), 24);
}

UTEST_CASE(tensor_close)
{
    constexpr auto epsilon = 1e-12;
    constexpr auto nan     = std::numeric_limits<double>::quiet_NaN();

    tensor_mem_t<double, 2> tensor0{};
    tensor_mem_t<double, 2> tensor1(10, 2);
    tensor_mem_t<double, 2> tensor2(10, 3);

    UTEST_CHECK(close(tensor0, tensor0, epsilon));
    UTEST_CHECK(!close(tensor0, tensor1, epsilon));
    UTEST_CHECK(!close(tensor0, tensor2, epsilon));

    UTEST_CHECK(close(tensor0.vector(), tensor0.vector(), epsilon));
    UTEST_CHECK(!close(tensor0.vector(), tensor1.vector(), epsilon));
    UTEST_CHECK(!close(tensor0.vector(), tensor2.vector(), epsilon));

    UTEST_CHECK(!close(tensor1, tensor2, epsilon));
    UTEST_CHECK(!close(tensor1.vector(), tensor2.vector(), epsilon));
    tensor2.resize(tensor1.dims());

    tensor1.zero();
    tensor2.zero();
    UTEST_CHECK(close(tensor1, tensor2, epsilon));
    UTEST_CHECK(close(tensor1.vector(), tensor2.vector(), epsilon));

    tensor1(11) = 11.0;
    UTEST_CHECK(!close(tensor1, tensor2, epsilon));
    UTEST_CHECK(!close(tensor1.vector(), tensor2.vector(), epsilon));

    tensor1(11) = nan;
    UTEST_CHECK(!close(tensor1, tensor2, epsilon));

    tensor2(11) = nan;
    UTEST_CHECK(close(tensor1, tensor2, epsilon));

    tensor1(7) = 42.42;
    tensor2(7) = 42.42 + 1e-15;
    UTEST_CHECK(close(tensor1, tensor2, epsilon));
}

UTEST_CASE(is_tensor)
{
    static_assert(is_tensor_v<tensor_mem_t<double, 1>>);
    static_assert(is_tensor_v<tensor_map_t<double, 2>>);
    static_assert(is_tensor_v<tensor_cmap_t<double, 3>>);

    static_assert(!is_tensor_v<int>);
    static_assert(!is_tensor_v<double>);
    static_assert(!is_tensor_v<std::vector<int>>);
}

UTEST_CASE(make_indices)
{
    const auto indices = make_indices(10, 42, 13);

    UTEST_CHECK_EQUAL(indices.size(), 3);
    UTEST_CHECK_EQUAL(indices(0), 10);
    UTEST_CHECK_EQUAL(indices(1), 42);
    UTEST_CHECK_EQUAL(indices(2), 13);
}

UTEST_CASE(make_full_tensor)
{
    const auto tensor = make_full_tensor<int>(make_dims(2, 3), 42);

    UTEST_CHECK_EQUAL(tensor.dims(), make_dims(2, 3));
    UTEST_CHECK_EQUAL(tensor(0, 0), 42);
    UTEST_CHECK_EQUAL(tensor(0, 1), 42);
    UTEST_CHECK_EQUAL(tensor(0, 2), 42);
    UTEST_CHECK_EQUAL(tensor(1, 0), 42);
    UTEST_CHECK_EQUAL(tensor(1, 1), 42);
    UTEST_CHECK_EQUAL(tensor(1, 2), 42);
}

UTEST_CASE(make_full_vector)
{
    const auto vector = make_full_vector<int>(5, 42);

    UTEST_CHECK_EQUAL(vector.size(), 5);
    UTEST_CHECK_EQUAL(vector(0), 42);
    UTEST_CHECK_EQUAL(vector(1), 42);
    UTEST_CHECK_EQUAL(vector(2), 42);
    UTEST_CHECK_EQUAL(vector(3), 42);
    UTEST_CHECK_EQUAL(vector(4), 42);
}

UTEST_CASE(make_full_matrix)
{
    const auto matrix = make_full_matrix<int>(3, 2, 42);

    UTEST_CHECK_EQUAL(matrix.rows(), 3);
    UTEST_CHECK_EQUAL(matrix.cols(), 2);
    UTEST_CHECK_EQUAL(matrix(0, 0), 42);
    UTEST_CHECK_EQUAL(matrix(0, 1), 42);
    UTEST_CHECK_EQUAL(matrix(1, 0), 42);
    UTEST_CHECK_EQUAL(matrix(1, 1), 42);
    UTEST_CHECK_EQUAL(matrix(2, 0), 42);
    UTEST_CHECK_EQUAL(matrix(2, 1), 42);
}

UTEST_CASE(mem_from_map)
{
    {
        auto tensor1 = make_full_tensor<int>(make_dims(2, 1), 42);

        tensor_mem_t<int, 2> tensor2 = tensor1.tensor();
        UTEST_CHECK_EQUAL(tensor2.dims(), make_dims(2, 1));
        UTEST_CHECK_EQUAL(tensor2(0, 0), 42);
        UTEST_CHECK_EQUAL(tensor2(1, 0), 42);

        tensor1(1, 0) = 17;
        UTEST_CHECK_EQUAL(tensor1(0, 0), 42);
        UTEST_CHECK_EQUAL(tensor1(1, 0), 17);
        UTEST_CHECK_EQUAL(tensor2(0, 0), 42);
        UTEST_CHECK_EQUAL(tensor2(1, 0), 42);
    }
    {
        const auto tensor1 = make_full_tensor<int>(make_dims(2, 1), 42);

        tensor_mem_t<int, 2> tensor2 = tensor1.tensor();
        UTEST_CHECK_EQUAL(tensor2.dims(), make_dims(2, 1));
        UTEST_CHECK_EQUAL(tensor2(0, 0), 42);
        UTEST_CHECK_EQUAL(tensor2(1, 0), 42);

        tensor2(1, 0) = 17;
        UTEST_CHECK_EQUAL(tensor2(0, 0), 42);
        UTEST_CHECK_EQUAL(tensor2(1, 0), 17);
        UTEST_CHECK_EQUAL(tensor1(0, 0), 42);
        UTEST_CHECK_EQUAL(tensor1(1, 0), 42);
    }
}

UTEST_CASE(make_random)
{
    for (auto trial = 0; trial < 100; ++trial)
    {
        const auto min = trial - 1;
        const auto max = trial + 7;

        const auto tensor = make_random_tensor<double>(make_dims(3, 4), min, max);
        UTEST_CHECK_EQUAL(tensor.dims(), make_dims(3, 4));
        UTEST_CHECK_LESS_EQUAL(min, tensor.min());
        UTEST_CHECK_LESS_EQUAL(tensor.max(), max);

        const auto vector = make_random_vector<double>(100, min, max);
        UTEST_CHECK_EQUAL(vector.size(), 100);
        UTEST_CHECK_LESS_EQUAL(min, vector.min());
        UTEST_CHECK_LESS_EQUAL(vector.max(), max);

        const auto matrix = make_random_matrix<double>(10, 11, min, max);
        UTEST_CHECK_EQUAL(matrix.rows(), 10);
        UTEST_CHECK_EQUAL(matrix.cols(), 11);
        UTEST_CHECK_LESS_EQUAL(min, matrix.min());
        UTEST_CHECK_LESS_EQUAL(matrix.max(), max);
    }
}

UTEST_CASE(make_random_fixed_seed)
{
    const auto min = -1.0;
    const auto max = +1.0;

    const auto tensor0 = make_random_tensor<double>(make_dims(3, 4), min, max, seed_t{});
    const auto tensor1 = make_random_tensor<double>(make_dims(3, 4), min, max, seed_t{42U});
    const auto tensor2 = make_random_tensor<double>(make_dims(3, 4), min, max, seed_t{42U});
    UTEST_CHECK_NOT_EQUAL(tensor0, tensor1);
    UTEST_CHECK_EQUAL(tensor1, tensor2);

    const auto vector0 = make_random_vector<double>(100, min, max, seed_t{});
    const auto vector1 = make_random_vector<double>(100, min, max, seed_t{17U});
    const auto vector2 = make_random_vector<double>(100, min, max, seed_t{17U});
    UTEST_CHECK_NOT_EQUAL(vector0, vector1);
    UTEST_CHECK_EQUAL(vector1, vector2);

    const auto matrix0 = make_random_matrix<double>(10, 11, min, max, seed_t{});
    const auto matrix1 = make_random_matrix<double>(10, 11, min, max, seed_t{11U});
    const auto matrix2 = make_random_matrix<double>(10, 11, min, max, seed_t{11U});
    UTEST_CHECK_NOT_EQUAL(matrix0, matrix1);
    UTEST_CHECK_EQUAL(matrix1, matrix2);
}

UTEST_END_MODULE()
