#include <nano/program/linear.h>
#include <utest/utest.h>

using namespace nano;
using namespace nano::program;

UTEST_BEGIN_MODULE(test_program_constraint)

UTEST_CASE(equality)
{
    {
        const auto constraint = equality_t<matrix_t, vector_t>{};
        UTEST_CHECK(!constraint.valid());
    }
    {
        const auto A          = make_matrix<scalar_t>(2, 2, 1, 0, 0, 1, 1);
        const auto b          = make_vector<scalar_t>(3, 2);
        const auto constraint = make_equality(A, b);
        UTEST_CHECK(constraint.valid());
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1.5, 0, 2)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(1, 1, 0)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(0, 1, 1)));
    }
}

UTEST_CASE(inequality)
{
    {
        const auto constraint = inequality_t<matrix_t, vector_t>{};
        UTEST_CHECK(!constraint.valid());
    }
    {
        const auto A          = make_matrix<scalar_t>(2, 2, 1, 0, 0, 1, 1);
        const auto b          = make_vector<scalar_t>(3, 2);
        const auto constraint = make_inequality(A, b);
        UTEST_CHECK(constraint.valid());
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1.5, 0, 2)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 0)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(0, 1, 1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(2, 1, 1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(1, 1, 2)));
    }
    {
        const auto upper      = make_vector<scalar_t>(+1, +1, +2);
        const auto constraint = make_less(upper);
        UTEST_CHECK(constraint.valid());
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(-1, -1, 2)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(0, 1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 2)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(1.1, 1, 1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(1, 1, 2.1)));
    }
    {
        const auto constraint = make_less(3, 1);
        UTEST_CHECK(constraint.valid());
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(-1, -1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(0, 1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(1, 1, 2)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(1.1, 1, 1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(1, 1, 1.1)));
    }
    {
        const auto lower      = make_vector<scalar_t>(-1, -1, -1);
        const auto constraint = make_greater(lower);
        UTEST_CHECK(constraint.valid());
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(-1, -1, 2)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(0, 1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(-1.1, -1, -1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(-1, -2, -3)));
    }
    {
        const auto constraint = make_greater(3, 1);
        UTEST_CHECK(constraint.valid());
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 2, 3)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(0, 1, 1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(0, 0, 0)));
    }
}

UTEST_CASE(inequality_strictly_feasible)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        for (const tensor_size_t ineqs : {dims - 1, dims, dims + 1, dims * 2})
        {
            for (auto test = 0; test < 100; ++test)
            {
                const auto c       = make_random_vector<scalar_t>(dims);
                const auto A       = make_random_matrix<scalar_t>(ineqs, dims);
                const auto b       = make_random_vector<scalar_t>(ineqs);
                const auto program = make_linear(c, make_inequality(A, b));

                const auto opt_x = program.make_strictly_feasible();

                // NB: it is guaranteed to always have a feasible point!
                if (ineqs <= dims)
                {
                    UTEST_REQUIRE(opt_x);
                    UTEST_CHECK_LESS((A * opt_x.value() - b).maxCoeff(), 0.0);
                }

                // NB: some random hyper-plane splits may not always have a feasible point!
                else if (opt_x)
                {
                    UTEST_CHECK_LESS((A * opt_x.value() - b).maxCoeff(), 0.0);
                }
            }
        }
    }
}

UTEST_CASE(inequality_strictly_feasible_bundle)
{
    // NB: generating a strictly feasible point fails for the FPBA solvers generated for the `chained_cb3I[4D]` problem.
    const auto A = make_matrix<scalar_t>(
        5, -13.0791713343359675, 11.0223780863932728, -4.4019980261743887, -2.5763086376600111, -1.0000000000000000,
        7215.0982713243365652, -9299047.8599894158542156, 9299623.7717038244009018, 6.5763086376600093,
        -1.0000000000000000, 7214.4055358504474498, -3412548.2061092313379049, 3412971.5455180155113339,
        6.5763076510207092, -1.0000000000000000, 7211.3160869768435077, -1247120.8129310656804591,
        1247420.0596358175389469, 6.5763032495401736, -1.0000000000000000, 7199.5211198816032265,
        -450621.4467068934463896, 450821.7497615875909105, 6.5762864247748309, -1.0000000000000000);

    const auto b = make_vector<scalar_t>(-1.4491983618949895, 133530540.3222339451313019, 45624197.2596000581979752,
                                         15460162.1538065522909164, 5169566.8448949512094259);

    const auto c = make_random_vector<scalar_t>(5);

    const auto program = make_linear(c, make_inequality(A, b));

    const auto opt_x = program.make_strictly_feasible();
    UTEST_REQUIRE(opt_x);
    UTEST_CHECK_LESS((A * opt_x.value() - b).maxCoeff(), 0.0);
}

UTEST_CASE(convex_hull_feasible_center)
{
    for (tensor_size_t dims = 2; dims < 100; dims += 3)
    {
        const auto c = vector_t::constant(dims, 1.42);

        const auto lower = program::make_less(dims, 1.0);
        const auto upper = program::make_greater(dims, 0.0);
        const auto wsum1 = program::make_equality(vector_t::constant(dims, 1.0), 1.0);

        const auto program = program::make_linear(c, lower, upper, wsum1);

        const auto x0 = vector_t{vector_t::constant(dims, 1.0 / static_cast<scalar_t>(dims))};
        UTEST_CHECK_LESS(program.m_eq.deviation(x0), epsilon0<scalar_t>());
        UTEST_CHECK_LESS(program.m_ineq.deviation(x0), epsilon0<scalar_t>());
        UTEST_CHECK(program.feasible(x0));
    }
}

UTEST_END_MODULE()
