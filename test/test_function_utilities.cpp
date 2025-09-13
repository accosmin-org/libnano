#include <fixture/function.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>
#include <nano/function/lambda.h>
#include <nano/function/util.h>
#include <nano/tensor/stack.h>

using namespace nano;

namespace
{
scalar_t lambda(const vector_cmap_t x, vector_map_t gx, matrix_map_t Hx)
{
    if (gx.size() == x.size())
    {
        gx = x;
    }

    if (Hx.rows() == x.size() && Hx.cols() == x.size())
    {
        Hx = matrix_t::identity(x.size(), x.size());
    }

    return 0.5 * x.dot(x);
}
} // namespace

UTEST_BEGIN_MODULE()

UTEST_CASE(make_full_rank)
{
    {
        auto A = matrix_t{};
        auto b = vector_t{};

        const auto stats = make_full_rank(A, b);
        UTEST_CHECK_EQUAL(stats.m_rank, 0);
        UTEST_CHECK_EQUAL(stats.m_changed, false);
    }

    for (const tensor_size_t dims : {3, 7, 11})
    {
        const auto D = make_random_tensor<scalar_t>(make_dims(2 * dims, dims));
        const auto Q = matrix_t{D.transpose() * D + 0.1 * matrix_t::identity(dims, dims)};
        const auto x = make_random_tensor<scalar_t>(make_dims(dims));

        // full rank
        {
            auto A = matrix_t{Q};
            auto b = vector_t{Q * x};

            const auto expected_A = A;
            const auto expected_b = b;

            const auto stats = make_full_rank(A, b);
            UTEST_CHECK_EQUAL(stats.m_rank, dims);
            UTEST_CHECK_EQUAL(stats.m_changed, false);
            UTEST_CHECK_CLOSE(A, expected_A, epsilon0<scalar_t>());
            UTEST_CHECK_CLOSE(b, expected_b, epsilon0<scalar_t>());
            UTEST_CHECK_CLOSE(vector_t{A * x}, b, 1e-15);
        }

        // duplicated rows
        {
            auto A = ::nano::stack<scalar_t>(2 * dims, dims, Q, Q);
            auto b = ::nano::stack<scalar_t>(2 * dims, Q * x, Q * x);

            const auto stats = make_full_rank(A, b);
            UTEST_CHECK_EQUAL(stats.m_rank, dims);
            UTEST_CHECK_EQUAL(stats.m_changed, true);
            UTEST_CHECK_EQUAL(A.rows(), dims);
            UTEST_CHECK_EQUAL(b.size(), dims);
            UTEST_CHECK_CLOSE(vector_t{A * x}, b, 1e-14);
        }

        // linear dependency
        {
            auto A = ::nano::stack<scalar_t>(2 * dims, dims, Q, 2.0 * Q);
            auto b = ::nano::stack<scalar_t>(2 * dims, Q * x, 2.0 * (Q * x));

            const auto stats = make_full_rank(A, b);
            UTEST_CHECK_EQUAL(stats.m_rank, dims);
            UTEST_CHECK_EQUAL(stats.m_changed, true);
            UTEST_CHECK_EQUAL(A.rows(), dims);
            UTEST_CHECK_EQUAL(b.size(), dims);
            UTEST_CHECK_CLOSE(vector_t{A * x}, b, 1e-14);
        }
    }
}

UTEST_CASE(remove_zero_rows_none)
{
    // clang-format off
    const auto A = make_matrix<scalar_t>(5,
        0.1, 0.0, 0.0,
        0.0, 0.2, 0.8,
        0.2, -0.1, 1.0,
        0.0, 0.0, 0.1,
        -1.0, -10.0, 0.0);
    const auto b = make_vector<scalar_t>(1.0, 1.0, 1.0, 1.0, 1.0);
    // clang-format on

    {
        auto Ax = A;
        auto bx = b;

        const auto stats = remove_zero_rows_equality(Ax, bx);
        UTEST_CHECK_EQUAL(stats.m_removed, 0);
        UTEST_CHECK_EQUAL(stats.m_inconsistent, 0);
        UTEST_CHECK_CLOSE(Ax, A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(bx, b, epsilon0<scalar_t>());
    }
    {
        auto Ax = A;
        auto bx = b;

        const auto stats = remove_zero_rows_inequality(Ax, bx);
        UTEST_CHECK_EQUAL(stats.m_removed, 0);
        UTEST_CHECK_EQUAL(stats.m_inconsistent, 0);
        UTEST_CHECK_CLOSE(Ax, A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(bx, b, epsilon0<scalar_t>());
    }
}

UTEST_CASE(remove_zero_rows_some)
{
    // clang-format off
    const auto A = make_matrix<scalar_t>(5,
        0.0, 0.0, 0.0,
        0.0, 0.2, 0.8,
        0.2, -0.1, 1.0,
        0.0, 0.0, 0.0,
        -1.0, -10.0, 0.0);
    const auto b1 = make_vector<scalar_t>(1.0, 1.0, 2.0, 1.0, 3.0);
    const auto b2 = make_vector<scalar_t>(0.0, 1.0, 2.0, -1.0, 3.0);
    const auto b3 = make_vector<scalar_t>(0.0, 1.0, 2.0, 0.0, 3.0);

    const auto expected_A = make_matrix<scalar_t>(3,
        0.0, 0.2, 0.8,
        0.2, -0.1, 1.0,
        -1.0, -10.0, 0.0);
    const auto expected_b = make_vector<scalar_t>(1.0, 2.0, 3.0);
    // clang-format on

    {
        auto Ax = A;
        auto bx = b1;

        const auto stats = remove_zero_rows_equality(Ax, bx);
        UTEST_CHECK_EQUAL(stats.m_removed, 2);
        UTEST_CHECK_EQUAL(stats.m_inconsistent, 2);
        UTEST_CHECK_CLOSE(Ax, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(bx, expected_b, epsilon0<scalar_t>());
    }
    {
        auto Ax = A;
        auto bx = b2;

        const auto stats = remove_zero_rows_equality(Ax, bx);
        UTEST_CHECK_EQUAL(stats.m_removed, 2);
        UTEST_CHECK_EQUAL(stats.m_inconsistent, 1);
        UTEST_CHECK_CLOSE(Ax, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(bx, expected_b, epsilon0<scalar_t>());
    }
    {
        auto Ax = A;
        auto bx = b3;

        const auto stats = remove_zero_rows_equality(Ax, bx);
        UTEST_CHECK_EQUAL(stats.m_removed, 2);
        UTEST_CHECK_EQUAL(stats.m_inconsistent, 0);
        UTEST_CHECK_CLOSE(Ax, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(bx, expected_b, epsilon0<scalar_t>());
    }
    {
        auto Ax = A;
        auto bx = b1;

        const auto stats = remove_zero_rows_inequality(Ax, bx);
        UTEST_CHECK_EQUAL(stats.m_removed, 2);
        UTEST_CHECK_EQUAL(stats.m_inconsistent, 0);
        UTEST_CHECK_CLOSE(Ax, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(bx, expected_b, epsilon0<scalar_t>());
    }
    {
        auto Ax = A;
        auto bx = b2;

        const auto stats = remove_zero_rows_inequality(Ax, bx);
        UTEST_CHECK_EQUAL(stats.m_removed, 2);
        UTEST_CHECK_EQUAL(stats.m_inconsistent, 1);
        UTEST_CHECK_CLOSE(Ax, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(bx, expected_b, epsilon0<scalar_t>());
    }
    {
        auto Ax = A;
        auto bx = b3;

        const auto stats = remove_zero_rows_inequality(Ax, bx);
        UTEST_CHECK_EQUAL(stats.m_removed, 2);
        UTEST_CHECK_EQUAL(stats.m_inconsistent, 0);
        UTEST_CHECK_CLOSE(Ax, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(bx, expected_b, epsilon0<scalar_t>());
    }
}

UTEST_CASE(is_convex)
{
    for (const tensor_size_t dims : {3, 7, 11})
    {
        auto Q = matrix_t{matrix_t::identity(dims, dims)};

        UTEST_CHECK(is_convex(Q));
        UTEST_CHECK_CLOSE(strong_convexity(Q), 1.0, epsilon0<scalar_t>());

        UTEST_CHECK(is_convex(2.0 * Q));
        UTEST_CHECK_CLOSE(strong_convexity(Q), 1.0, epsilon0<scalar_t>());

        Q(0, 0) = -1.0;
        UTEST_CHECK(!is_convex(Q));
        UTEST_CHECK_CLOSE(strong_convexity(Q), 0.0, epsilon0<scalar_t>());

        Q = matrix_t::zero(dims, dims);
        UTEST_CHECK(is_convex(Q));

        Q = -matrix_t::identity(dims, dims);
        UTEST_CHECK(!is_convex(Q));

        const auto D = make_random_matrix<scalar_t>(dims, dims);
        Q            = D.transpose() * D;
        UTEST_CHECK(is_convex(Q));

        Q = D.transpose() * D + matrix_t::identity(dims, dims);
        UTEST_CHECK(is_convex(Q));

        Q = -D.transpose() * D - matrix_t::identity(dims, dims);
        UTEST_CHECK(!is_convex(Q));

        Q = matrix_t::identity(dims, dims);
        Q(0, 1) += 1.0;
        UTEST_CHECK(!is_convex(Q));
    }
}

UTEST_CASE(is_convex_matrixD)
{
    for (const tensor_size_t rows : {3, 7, 11})
    {
        for (const tensor_size_t cols : {rows / 2, rows - 1, rows, rows + 1, 2 * rows})
        {
            const auto D = make_random_matrix<scalar_t>(rows, cols);
            UTEST_CHECK(is_convex(D * D.transpose()));
            UTEST_CHECK(is_convex(D.transpose() * D));
        }
    }
}

UTEST_CASE(is_convex_matrixG1)
{
    // NB: use case generated by the gradient sampling solver
    // clang-format off
    const auto G = make_matrix<scalar_t>(9,
        -2.9906464007632385, 0.1845195874589916, -3.5083435977220434, -2.8884348992822542,
        -3.1918966653654079, 0.4102772726563952, -3.7181087399161696, -2.6823288434056969,
        -3.0267542392872291, 0.1430742734183924, -3.4348754887267989, -2.8994101187082277,
        -3.1499273511543615, 0.3931582930223101, -3.6922990402444849, -2.7596079237635216,
        -3.2108515441517529, 0.2210473007944523, -3.3997240574469840, -2.8874864412375123,
        -3.0709898122812347, 0.3285203666535044, -3.5734370533306201, -2.9460349501918524,
        -3.0793025727414731, 0.2648734324672459, -3.4581729746899557, -2.9482660417643940,
        -3.1258424062149262, 0.2583750562363925, -3.6398541627615542, -2.7166684692573213,
        -3.1235785286248761, 0.2679528496227962, -3.5424406718150632, -2.8479180082365847);
    // clang-format on
    UTEST_CHECK(is_convex(G * G.transpose()));
    UTEST_CHECK(is_convex(G.transpose() * G));
}

UTEST_CASE(is_convex_matrixG2)
{
    // NB: use case generated by the gradient sampling solver
    // clang-format off
    const auto G = make_matrix<scalar_t>(9,
        4627.6630249405197901, 1071.9738384689760551, -6102.3684392326531452, 5172.6689123251844649,
        4586.9698298480880112, 1690.9009394084903306, -6476.6347129707082786, 4490.4160561144763051,
        5062.7069212585383866, 1092.0610305849625092, -6901.0934027438415796, 5073.2996150996486904,
        5476.6323050846422120, 1445.4509633042778205, -6344.7749355460055085, 4586.0272879818394358,
        5571.8218892965242048, 1972.5029917246854438, -6492.8927355842488396, 5457.9361252145436083,
        4421.8160667291176651, 1937.8110813395976493, -7165.8052651027928732, 5062.4482186546729281,
        4869.0457396402853192, 1891.6866969960226470, -6625.9523963654901308, 4412.1502361305865634,
        5310.0177617526869653, 1784.9329761301944473, -7052.0990550076849104, 4628.1763730168422626,
        4947.7878264535438575, 1724.3588414530079262, -6416.7417392552788442, 5292.4661056881668628);
    // clang-format on
    UTEST_CHECK(is_convex(G * G.transpose()));
    UTEST_CHECK(is_convex(G.transpose() * G));
}

UTEST_CASE(make_linear_constraints)
{
    auto function = make_function(3, convexity::yes, smoothness::yes, 2.0, lambda);
    {
        const auto lconstraints = make_linear_constraints(function);
        UTEST_REQUIRE(lconstraints.has_value());

        const auto& [A, b, G, h] = lconstraints.value();
        const auto expected_A    = matrix_t{0, 3};
        const auto expected_b    = vector_t{0};
        const auto expected_G    = matrix_t{0, 3};
        const auto expected_h    = vector_t{0};

        UTEST_CHECK_CLOSE(A, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(b, expected_b, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(G, expected_G, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(h, expected_h, epsilon0<scalar_t>());
    }
    critical(function.variable() >= 2.0);
    {
        const auto lconstraints = make_linear_constraints(function);
        UTEST_REQUIRE(lconstraints.has_value());

        const auto& [A, b, G, h] = lconstraints.value();
        const auto expected_A    = matrix_t{0, 3};
        const auto expected_b    = vector_t{0};
        // clang-format off
        const auto expected_G    = make_tensor<scalar_t>(
            make_dims(3, 3),
            -1, 0, 0,
            0, -1, 0,
            0, 0, -1);
        const auto expected_h    = make_vector<scalar_t>(
            -2,
            -2,
            -2);
        // clang-format on

        UTEST_CHECK_CLOSE(A, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(b, expected_b, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(G, expected_G, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(h, expected_h, epsilon0<scalar_t>());
    }
    critical(function.variable() <= 3.7);
    {
        const auto lconstraints = make_linear_constraints(function);
        UTEST_REQUIRE(lconstraints.has_value());

        const auto& [A, b, G, h] = lconstraints.value();
        const auto expected_A    = matrix_t{0, 3};
        const auto expected_b    = vector_t{0};
        // clang-format off
        const auto expected_G    = make_tensor<scalar_t>(
            make_dims(6, 3),
            -1, +0, +0,
            +0, -1, +0,
            +0, +0, -1,
            +1, +0, +0,
            +0, +1, +0,
            +0, +0, +1);
        const auto expected_h    = make_vector<scalar_t>(
            -2.0,
            -2.0,
            -2.0,
            +3.7,
            +3.7,
            +3.7);
        // clang-format on

        UTEST_CHECK_CLOSE(A, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(b, expected_b, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(G, expected_G, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(h, expected_h, epsilon0<scalar_t>());
    }
    critical(vector_t::constant(3, 1.0) * function.variable() == 12.0);
    {
        const auto lconstraints = make_linear_constraints(function);
        UTEST_REQUIRE(lconstraints.has_value());

        const auto& [A, b, G, h] = lconstraints.value();
        // clang-format off
        const auto expected_A    = make_tensor<scalar_t>(
            make_dims(1, 3),
            +1, +1, +1);
        const auto expected_b    = make_vector<scalar_t>(
            12.0);
        const auto expected_G    = make_tensor<scalar_t>(
            make_dims(6, 3),
            -1, +0, +0,
            +0, -1, +0,
            +0, +0, -1,
            +1, +0, +0,
            +0, +1, +0,
            +0, +0, +1);
        const auto expected_h    = make_vector<scalar_t>(
            -2.0,
            -2.0,
            -2.0,
            +3.7,
            +3.7,
            +3.7);
        // clang-format on

        UTEST_CHECK_CLOSE(A, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(b, expected_b, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(G, expected_G, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(h, expected_h, epsilon0<scalar_t>());
    }
    UTEST_REQUIRE(
        function.constrain(constraint::euclidean_ball_equality_t{make_vector<scalar_t>(0.0, 0.0, 0.0), 30.0}));
    {
        const auto lconstraints = make_linear_constraints(function);
        UTEST_REQUIRE(!lconstraints.has_value());
    }
}

UTEST_END_MODULE()
