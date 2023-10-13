#include <nano/program/linear.h>
#include <utest/utest.h>

using namespace nano;
using namespace nano::program;

UTEST_BEGIN_MODULE(test_program_constraint)

UTEST_CASE(equality)
{
    {
        const auto constraint = equality_t{};
        UTEST_CHECK(!constraint);
    }
    {
        const auto constraint = equality_t{} & equality_t{};
        UTEST_CHECK(!constraint);
    }
    {
        const auto A          = make_matrix<scalar_t>(2, 2, 1, 0, 0, 1, 1);
        const auto b          = make_vector<scalar_t>(3, 2);
        const auto constraint = equality_t{} & equality_t{A, b};
        UTEST_CHECK(constraint);
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1.5, 0, 2)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(1, 1, 0)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(0, 1, 1)));
    }
}

UTEST_CASE(inequality)
{
    {
        const auto constraint = inequality_t{};
        UTEST_CHECK(!constraint);
    }
    {
        const auto constraint = inequality_t{} & inequality_t{};
        UTEST_CHECK(!constraint);
    }
    {
        const auto A          = make_matrix<scalar_t>(2, 2, 1, 0, 0, 1, 1);
        const auto b          = make_vector<scalar_t>(3, 2);
        const auto constraint = inequality_t{} & inequality_t{A, b};
        UTEST_CHECK(constraint);
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1.5, 0, 2)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 0)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(0, 1, 1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(2, 1, 1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(1, 1, 2)));
    }
    {
        const auto lower      = make_vector<scalar_t>(-1, -1, -1);
        const auto upper      = make_vector<scalar_t>(+1, +1, +2);
        const auto constraint = inequality_t::from_rectangle(lower, upper) & inequality_t{};
        UTEST_CHECK(constraint);
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(-1, -1, 2)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(0, 1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 2)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(1.1, 1, 1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(1, 1, -1.1)));
    }
    {
        const auto lower      = make_vector<scalar_t>(-1, -1, -1);
        const auto constraint = inequality_t::greater(lower);
        UTEST_CHECK(constraint);
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(-1, -1, 2)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(0, 1, 1)));
        UTEST_CHECK(constraint.feasible(make_vector<scalar_t>(1, 1, 1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(-1.1, -1, -1)));
        UTEST_CHECK(!constraint.feasible(make_vector<scalar_t>(-1, -2, -3)));
    }
    {
        const auto constraint = inequality_t::greater(3, 1);
        UTEST_CHECK(constraint);
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
                const auto A          = make_random_matrix<scalar_t>(ineqs, dims);
                const auto b          = make_random_vector<scalar_t>(ineqs);
                const auto inequality = inequality_t{A, b};

                const auto opt_x = inequality.make_strictly_feasible();

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

UTEST_END_MODULE()
