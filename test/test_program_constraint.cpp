#include <nano/function/linear.h>
#include <nano/function/quadratic.h>
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

UTEST_CASE(convex_hull_feasible_center)
{
    for (tensor_size_t dims = 2; dims < 100; dims += 3)
    {
        const auto c = vector_t::constant(dims, 1.42);

        const auto lower = program::make_less(dims, 1.0);
        const auto upper = program::make_greater(dims, 0.0);
        const auto wsum1 = program::make_equality(vector_t::constant(dims, 1.0), 1.0);

        const auto program = program::make_linear(c, lower, upper, wsum1);
        const auto epsilon = 5.0 * epsilon0<scalar_t>();

        const auto x0 = vector_t{vector_t::constant(dims, 1.0 / static_cast<scalar_t>(dims))};
        UTEST_CHECK_LESS(program.m_eq.deviation(x0), epsilon);
        UTEST_CHECK_LESS(program.m_ineq.deviation(x0), epsilon);
        UTEST_CHECK(program.feasible(x0, epsilon));
    }
}

UTEST_END_MODULE()
