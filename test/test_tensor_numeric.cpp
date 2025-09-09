#include <nano/tensor/tensor.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE()

UTEST_CASE(from_eigen_vector_expression)
{
    using ttensor_t = tensor_mem_t<int, 1>;

    const auto tensor0 = ttensor_t{ttensor_t::zero(5)};
    const auto tensorC = ttensor_t{ttensor_t::constant(6, 2)};

    UTEST_CHECK_EQUAL(tensor0, make_tensor<int>(make_dims(5), 0, 0, 0, 0, 0));
    UTEST_CHECK_EQUAL(tensorC, make_tensor<int>(make_dims(6), 2, 2, 2, 2, 2, 2));

    // also check assignment from expression
    auto tensorX = ttensor_t{};
    tensorX      = ttensor_t::zero(4);
    UTEST_CHECK_EQUAL(tensorX, make_tensor<int>(make_dims(4), 0, 0, 0, 0));

    // also check assignment
    tensorX = tensorC;
    UTEST_CHECK_EQUAL(tensorX, make_tensor<int>(make_dims(6), 2, 2, 2, 2, 2, 2));

    // also check copy construction
    const auto tensorY = tensorX;
    const auto tensorZ = ttensor_t{std::move(tensorX)};

    UTEST_CHECK_EQUAL(tensorY, make_tensor<int>(make_dims(6), 2, 2, 2, 2, 2, 2));
    UTEST_CHECK_EQUAL(tensorZ, make_tensor<int>(make_dims(6), 2, 2, 2, 2, 2, 2));
}

UTEST_CASE(from_eigen_matrix_expression)
{
    using ttensor_t = tensor_mem_t<int, 2>;

    const auto tensor0 = ttensor_t{ttensor_t::zero(2, 3)};
    const auto tensorC = ttensor_t{ttensor_t::constant(4, 3, 1)};
    const auto tensorI = ttensor_t{ttensor_t::identity(3, 3)};

    UTEST_CHECK_EQUAL(tensor0, make_tensor<int>(make_dims(2, 3), 0, 0, 0, 0, 0, 0));
    UTEST_CHECK_EQUAL(tensorC, make_tensor<int>(make_dims(4, 3), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
    UTEST_CHECK_EQUAL(tensorI, make_tensor<int>(make_dims(3, 3), 1, 0, 0, 0, 1, 0, 0, 0, 1));
}

UTEST_CASE(vector_elementwise)
{
    using vector_t = tensor_mem_t<int, 1>;

    auto vector = vector_t{vector_t::zero(4)};

    vector += vector_t::constant(4, 1);
    UTEST_CHECK_EQUAL(vector, make_tensor<int>(make_dims(4), 1, 1, 1, 1));

    vector -= vector_t::constant(4, 1);
    UTEST_CHECK_EQUAL(vector, make_tensor<int>(make_dims(4), 0, 0, 0, 0));

    vector += vector_t{vector_t::constant(4, 2)};
    vector *= 3;
    UTEST_CHECK_EQUAL(vector, make_tensor<int>(make_dims(4), 6, 6, 6, 6));

    vector /= 6;
    UTEST_CHECK_EQUAL(vector, make_tensor<int>(make_dims(4), 1, 1, 1, 1));
}

UTEST_CASE(matrix_elemwise)
{
    using matrix_t = tensor_mem_t<int, 2>;

    auto matrix = matrix_t{matrix_t::identity(3, 3)};

    matrix += matrix_t::constant(3, 3, 2);
    UTEST_CHECK_EQUAL(matrix, make_tensor<int>(make_dims(3, 3), 3, 2, 2, 2, 3, 2, 2, 2, 3));

    matrix -= matrix_t::constant(3, 3, 1);
    UTEST_CHECK_EQUAL(matrix, make_tensor<int>(make_dims(3, 3), 2, 1, 1, 1, 2, 1, 1, 1, 2));

    matrix *= 2;
    UTEST_CHECK_EQUAL(matrix, make_tensor<int>(make_dims(3, 3), 4, 2, 2, 2, 4, 2, 2, 2, 4));

    matrix -= matrix_t{matrix_t::constant(3, 3, 2)};
    UTEST_CHECK_EQUAL(matrix, make_tensor<int>(make_dims(3, 3), 2, 0, 0, 0, 2, 0, 0, 0, 2));

    matrix += matrix_t{matrix_t::constant(3, 3, -1)};
    UTEST_CHECK_EQUAL(matrix, make_tensor<int>(make_dims(3, 3), +1, -1, -1, -1, +1, -1, -1, -1, +1));
}

UTEST_CASE(all_finite)
{
    using vector_t = tensor_mem_t<double, 1>;

    auto vector = vector_t{vector_t::zero(3)};
    UTEST_CHECK(vector.all_finite());

    vector(0) = std::numeric_limits<double>::infinity();
    UTEST_CHECK(!vector.all_finite());

    vector(0) = -4.2;
    UTEST_CHECK(vector.all_finite());

    vector(1) = std::numeric_limits<double>::quiet_NaN();
    UTEST_CHECK(!vector.all_finite());
}

UTEST_CASE(eigen_expressions)
{
    using vector_t = tensor_mem_t<int, 1>;
    using matrix_t = tensor_mem_t<int, 2>;

    auto matrix  = matrix_t{matrix_t::identity(3, 3)};
    matrix(0, 1) = 2;
    matrix(0, 2) = 3;
    matrix += matrix_t{matrix.transpose()};
    UTEST_CHECK_EQUAL(matrix, make_tensor<int>(make_dims(3, 3), 2, 2, 3, 2, 2, 0, 3, 0, 2));

    const auto row0 = vector_t{matrix.row(0)};
    const auto part = vector_t{matrix.reshape(-1).segment(3, 2)};

    UTEST_CHECK_EQUAL(row0, make_tensor<int>(make_dims(3), 2, 2, 3));
    UTEST_CHECK_EQUAL(part, make_tensor<int>(make_dims(2), 2, 2));

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const auto matrixT = matrix_t{const_cast<const matrix_t&>(matrix).transpose()};
    const auto matrixV = vector_t{matrixT.vector()};
    const auto column0 = vector_t{matrixT.row(0)};
    const auto column1 = vector_t{matrixT.row(1)};
    const auto column2 = vector_t{matrixT.row(2)};
    const auto segment = vector_t{matrixV.segment(4, 5)};

    UTEST_CHECK_EQUAL(matrixT, make_tensor<int>(make_dims(3, 3), 2, 2, 3, 2, 2, 0, 3, 0, 2));
    UTEST_CHECK_EQUAL(column0, make_tensor<int>(make_dims(3), 2, 2, 3));
    UTEST_CHECK_EQUAL(column1, make_tensor<int>(make_dims(3), 2, 2, 0));
    UTEST_CHECK_EQUAL(column2, make_tensor<int>(make_dims(3), 3, 0, 2));
    UTEST_CHECK_EQUAL(segment, make_tensor<int>(make_dims(5), 2, 0, 3, 0, 2));
}

UTEST_CASE(tensor_stats)
{
    using vector_t = tensor_mem_t<double, 1>;

    const auto indices = arange(0, 10);
    const auto values0 = vector_t{};
    const auto values1 = vector_t{vector_t::constant(1, 0.0)};
    const auto valuesN = vector_t{indices.array().cast<double>()};

    UTEST_CHECK_CLOSE(valuesN.min(), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(valuesN.max(), 9.0, 1e-15);
    UTEST_CHECK_CLOSE(valuesN.sum(), 45.0, 1e-15);
    UTEST_CHECK_CLOSE(valuesN.mean(), 4.5, 1e-15);
    UTEST_CHECK_CLOSE(valuesN.stdev(), 0.9574271077563381, 1e-15);

    UTEST_CHECK_CLOSE(values0.variance(), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(values1.variance(), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(valuesN.variance(), 8.25, 1e-15);

    UTEST_CHECK_CLOSE(valuesN.lpNorm<1>(), 45.0, 1e-15);
    UTEST_CHECK_CLOSE(valuesN.lpNorm<Eigen::Infinity>(), 9.0, 1e-15);

    UTEST_CHECK_CLOSE(valuesN.dot(valuesN), 285.0, 1e-15);
    UTEST_CHECK_CLOSE(valuesN.squaredNorm(), 285.0, 1e-15);
    UTEST_CHECK_CLOSE(valuesN.dot(valuesN.vector()), 285.0, 1e-15);
}

UTEST_CASE(operators)
{
    using vector_t = tensor_mem_t<int, 1>;
    using matrix_t = tensor_mem_t<int, 2>;

    {
        const auto v1 = vector_t{vector_t::constant(3, 1)};
        const auto v4 = vector_t{4 * v1};
        const auto v5 = vector_t{v1 * 5};
        const auto v2 = vector_t{v4 / 2};
        const auto vN = vector_t{-v1};
        const auto mv = vector_t{matrix_t::identity(3, 3) * v1};
        const auto v3 = vector_t{v1 + vector_t::constant(3, 2)};
        const auto v6 = vector_t{v3 + v3};
        const auto v7 = vector_t{vector_t::constant(3, 4) + v3};
        const auto v0 = vector_t{v1 - vector_t::constant(3, 1)};
        const auto v9 = vector_t{vector_t::constant(3, 10) - v1};
        const auto v8 = vector_t{v9 - v1};

        UTEST_CHECK_EQUAL(v0, make_tensor<int>(make_dims(3), 0, 0, 0));
        UTEST_CHECK_EQUAL(v1, make_tensor<int>(make_dims(3), 1, 1, 1));
        UTEST_CHECK_EQUAL(v2, make_tensor<int>(make_dims(3), 2, 2, 2));
        UTEST_CHECK_EQUAL(v3, make_tensor<int>(make_dims(3), 3, 3, 3));
        UTEST_CHECK_EQUAL(v4, make_tensor<int>(make_dims(3), 4, 4, 4));
        UTEST_CHECK_EQUAL(v5, make_tensor<int>(make_dims(3), 5, 5, 5));
        UTEST_CHECK_EQUAL(v6, make_tensor<int>(make_dims(3), 6, 6, 6));
        UTEST_CHECK_EQUAL(v7, make_tensor<int>(make_dims(3), 7, 7, 7));
        UTEST_CHECK_EQUAL(v8, make_tensor<int>(make_dims(3), 8, 8, 8));
        UTEST_CHECK_EQUAL(v9, make_tensor<int>(make_dims(3), 9, 9, 9));
        UTEST_CHECK_EQUAL(vN, make_tensor<int>(make_dims(3), -1, -1, -1));
        UTEST_CHECK_EQUAL(mv, make_tensor<int>(make_dims(3), 1, 1, 1));
    }
    {
        const auto v1 = vector_t{vector_t::constant(2, 1)};
        const auto m1 = matrix_t{matrix_t::identity(2, 2)};
        const auto m2 = matrix_t{2 * m1};
        const auto m3 = matrix_t{m1 * 3};
        const auto mM = matrix_t{m3 / 3};
        const auto mu = vector_t{m2 * v1};
        const auto m0 = matrix_t{m1 - mM};
        const auto m4 = matrix_t{m3 + matrix_t::identity(2, 2)};
        const auto m5 = matrix_t{m4 - (-1 * matrix_t::identity(2, 2))};
        const auto mv = vector_t{m2 * vector_t::constant(2, 1)};
        const auto mw = vector_t{matrix_t::identity(2, 2) * v1};
        const auto mm = matrix_t{matrix_t::identity(2, 2) * m3};

        UTEST_CHECK_EQUAL(m0, make_tensor<int>(make_dims(2, 2), 0, 0, 0, 0));
        UTEST_CHECK_EQUAL(m1, make_tensor<int>(make_dims(2, 2), 1, 0, 0, 1));
        UTEST_CHECK_EQUAL(m2, make_tensor<int>(make_dims(2, 2), 2, 0, 0, 2));
        UTEST_CHECK_EQUAL(m3, make_tensor<int>(make_dims(2, 2), 3, 0, 0, 3));
        UTEST_CHECK_EQUAL(m4, make_tensor<int>(make_dims(2, 2), 4, 0, 0, 4));
        UTEST_CHECK_EQUAL(m5, make_tensor<int>(make_dims(2, 2), 5, 0, 0, 5));
        UTEST_CHECK_EQUAL(mM, make_tensor<int>(make_dims(2, 2), 1, 0, 0, 1));
        UTEST_CHECK_EQUAL(mu, make_tensor<int>(make_dims(2), 2, 2));
        UTEST_CHECK_EQUAL(mv, make_tensor<int>(make_dims(2), 2, 2));
        UTEST_CHECK_EQUAL(mw, make_tensor<int>(make_dims(2), 1, 1));
        UTEST_CHECK_EQUAL(mm, make_tensor<int>(make_dims(2, 2), 3, 0, 0, 3));
    }
}

UTEST_END_MODULE()
