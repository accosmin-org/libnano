#include <Eigen/Dense>
#include <function/program/cvx.h>

using namespace nano;
using namespace nano::program;

namespace
{
auto make_expected(const scalar_t fbest)
{
    return expected_t{}.fbest(fbest);
}

auto make_expected(vector_t xbest, const linear_program_t& program)
{
    auto expected = expected_t{std::move(xbest)};
    expected.fbest(program.m_c.dot(expected.m_xbest));
    return expected;
}

auto make_expected(vector_t xbest, const quadratic_program_t& program)
{
    auto expected = expected_t{std::move(xbest)};
    expected.fbest(expected.m_xbest.dot(0.5 * program.m_Q * expected.m_xbest + program.m_c));
    return expected;
}

auto make_xbest_cvx48d(const vector_t& c)
{
    const auto dims = c.size();
    const auto cmin = c.min();

    auto count = 0.0;
    auto xbest = make_full_vector<scalar_t>(dims, 0.0);
    for (tensor_size_t i = 0; i < dims; ++i)
    {
        if (c(i) == cmin)
        {
            ++count;
            xbest(i) = 1.0;
        }
    }
    xbest.array() /= count;
    return xbest;
}

auto make_sorted_cvx48e(const vector_t& c)
{
    std::vector<std::pair<scalar_t, tensor_size_t>> values;
    values.reserve(static_cast<size_t>(c.size()));
    for (tensor_size_t i = 0; i < c.size(); ++i)
    {
        values.emplace_back(c(i), i);
    }
    std::sort(values.begin(), values.end());
    return values;
}

auto make_sorted_cvx48f(const vector_t& c, const vector_t& d)
{
    std::vector<std::pair<scalar_t, tensor_size_t>> values;
    values.reserve(static_cast<size_t>(c.size()));
    for (tensor_size_t i = 0; i < c.size(); ++i)
    {
        values.emplace_back(c(i) / d(i), i);
    }
    std::sort(values.begin(), values.end());
    return values;
}
} // namespace

linear_program_cvx48b_t::linear_program_cvx48b_t(const tensor_size_t dims, const scalar_t lambda)
    : linear_program_t(scat("cvx48-[lambda=", lambda, "]"), dims)
{
    assert(lambda <= 0.0);

    const auto a = make_random_vector<scalar_t>(dims, +1.0, +2.0);
    const auto b = urand<scalar_t>(-1.0, +1.0);
    const auto c = lambda * a;

    reset((c);
    (a * (*this)) <= b;

    xbest(lambda * b);
}

rfunction_t linear_program_cvx48b_t::clone() const
{
    return std::make_unique<linear_program_cvx48b_t>(*this);
}

rfunction_t linear_program_cvx48b_t::make(const tensor_size_t dims, [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx48b_t>(dims);
}

expected_linear_program_t nano::program::make_linear_program_cvx48c(const tensor_size_t dims)
{
    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto l = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto u = make_random_vector<scalar_t>(dims, +1.0, +3.0);

    auto program  = make_linear(c, make_greater(l), make_less(u));
    auto xbest    = l.array() * c.array().max(0.0).sign() - u.array() * c.array().min(0.0).sign();
    auto expected = make_expected(xbest, program);

    return std::make_tuple(std::move(program), std::move(expected));
}

expected_linear_program_t nano::program::make_linear_program_cvx48d_eq(const tensor_size_t dims)
{
    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto A = vector_t::constant(dims, 1.0);
    const auto b = 1.0;

    auto program  = make_linear(c, make_equality(A, b), make_greater(dims, 0.0));
    auto xbest    = make_xbest_cvx48d(c);
    auto expected = make_expected(std::move(xbest), program);

    return std::make_tuple(std::move(program), std::move(expected));
}

expected_linear_program_t nano::program::make_linear_program_cvx48d_ineq(const tensor_size_t dims)
{
    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto A = vector_t::constant(dims, 1.0);
    const auto N = -matrix_t::identity(dims, dims);
    const auto b = 1.0;
    const auto z = vector_t::constant(dims, 0.0);

    auto program  = make_linear(c, make_inequality(A, b), make_inequality(N, z), make_greater(dims, 0.0));
    auto xbest    = c.min() < 0.0 ? make_xbest_cvx48d(c) : make_full_vector<scalar_t>(dims, 0.0);
    auto expected = make_expected(std::move(xbest), program);

    return std::make_tuple(std::move(program), std::move(expected));
}

expected_linear_program_t nano::program::make_linear_program_cvx48e_eq(const tensor_size_t dims,
                                                                       const tensor_size_t alpha)
{
    assert(alpha >= 0);
    assert(alpha <= dims);

    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto a = make_full_vector<scalar_t>(dims, 1.0);
    const auto v = make_sorted_cvx48e(c);
    const auto h = static_cast<scalar_t>(alpha);

    auto program = make_linear(c, make_equality(a, h), make_greater(dims, 0.0), make_less(dims, 1.0));

    auto xbest = make_full_vector<scalar_t>(dims, 0.0);
    for (tensor_size_t i = 0; i < alpha; ++i)
    {
        const auto [value, index] = v[static_cast<size_t>(i)];
        xbest(index)              = 1.0;
    }
    auto expected = make_expected(std::move(xbest), program);

    return std::make_tuple(std::move(program), std::move(expected));
}

expected_linear_program_t nano::program::make_linear_program_cvx48e_ineq(const tensor_size_t dims,
                                                                         const tensor_size_t alpha)
{
    assert(alpha >= 0);
    assert(alpha <= dims);

    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto a = make_full_vector<scalar_t>(dims, 1.0);
    const auto v = make_sorted_cvx48e(c);
    const auto h = static_cast<scalar_t>(alpha);

    auto program = make_linear(c, make_inequality(a, h), make_greater(dims, 0.0), make_less(dims, 1.0));
    auto xbest   = make_full_vector<scalar_t>(dims, 0.0);
    for (tensor_size_t i = 0, count = 0; i < dims && count < alpha; ++i)
    {
        const auto [value, index] = v[static_cast<size_t>(i)];
        if (value <= 0.0)
        {
            ++count;
            xbest(index) = 1.0;
        }
    }
    auto expected = make_expected(std::move(xbest), program);
    expected.status(alpha == 0 ? solver_status::unfeasible : solver_status::converged);

    return std::make_tuple(std::move(program), std::move(expected));
}

expected_linear_program_t nano::program::make_linear_program_cvx48f(const tensor_size_t dims, scalar_t alpha)
{
    assert(alpha >= 0.0);
    assert(alpha <= 1.0);

    const auto d = make_random_vector<scalar_t>(dims, 1.0, +2.0);
    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto v = make_sorted_cvx48f(c, d);

    alpha = alpha * d.sum();

    auto program = make_linear(c, make_equality(d, alpha), make_greater(dims, 0.0), make_less(dims, 1.0));
    auto accum   = 0.0;
    auto xbest   = make_full_vector<scalar_t>(dims, 0.0);
    for (tensor_size_t i = 0; i < dims && accum < alpha; ++i)
    {
        [[maybe_unused]] const auto [value, index] = v[static_cast<size_t>(i)];
        if (accum + d(index) <= alpha)
        {
            xbest(index) = 1.0;
        }
        else
        {
            xbest(index) = (alpha - accum) / d(index);
        }
        accum += d(index);
    }
    auto expected = make_expected(std::move(xbest), program);

    return std::make_tuple(std::move(program), std::move(expected));
}

expected_linear_program_t nano::program::make_linear_program_cvx49(const tensor_size_t dims)
{
    const auto c = make_random_vector<scalar_t>(dims, -1.0, -0.0);
    const auto A = matrix_t::identity(dims, dims);
    const auto b = make_random_vector<scalar_t>(dims, -1.0, +1.0);

    auto program  = make_linear(c, make_inequality(A, b));
    auto expected = make_expected(b, program);

    return std::make_tuple(std::move(program), std::move(expected));
}

expected_linear_program_t nano::program::make_linear_program_cvx410(const tensor_size_t dims, const bool feasible)
{
    const auto D = make_random_matrix<scalar_t>(dims, dims);
    const auto A = D.transpose() * D + matrix_t::identity(dims, dims);
    const auto c = make_random_vector<scalar_t>(dims);

    if (feasible)
    {
        // the solution is feasible
        const auto x = make_random_vector<scalar_t>(dims, +1.0, +2.0);
        const auto b = A * x;

        auto program  = make_linear(c, make_equality(A, b), make_greater(dims, 0.0));
        auto expected = make_expected(x, program);

        return std::make_tuple(std::move(program), std::move(expected));
    }
    else
    {
        // the solution is not feasible
        const auto x = make_random_vector<scalar_t>(dims, -2.0, -1.0);
        const auto b = A * x;

        auto program  = make_linear(c, make_equality(A, b), make_greater(dims, 0.0));
        auto expected = make_expected(x, program);
        expected.status(solver_status::unfeasible);

        return std::make_tuple(std::move(program), std::move(expected));
    }
}

expected_quadratic_program_t nano::program::make_quadratic_program_numopt162(const tensor_size_t dims,
                                                                             const tensor_size_t neqs)
{
    assert(neqs >= 1);
    assert(neqs <= dims);

    const auto x0 = make_random_vector<scalar_t>(dims);
    const auto Q  = matrix_t::identity(dims, dims);
    const auto c  = -x0;

    auto L = make_random_matrix<scalar_t>(neqs, neqs);
    auto U = make_random_matrix<scalar_t>(neqs, dims);

    L.matrix().triangularView<Eigen::Upper>().setZero();
    U.matrix().triangularView<Eigen::Lower>().setZero();

    L.diagonal().array() = 1.0;
    U.diagonal().array() = 1.0;

    const auto A     = L * U;
    const auto b     = make_random_vector<scalar_t>(neqs);
    const auto invAA = (A * A.transpose()).inverse();
    const auto xbest = x0 + A.transpose() * invAA * (b - A * x0);

    auto program  = make_quadratic(Q, c, make_equality(A, b));
    auto expected = make_expected(xbest, program);

    return std::make_tuple(std::move(program), std::move(expected));
}

expected_quadratic_program_t nano::program::make_quadratic_program_numopt1625(const tensor_size_t dims)
{
    const auto x0 = make_random_vector<scalar_t>(dims);
    const auto Q  = matrix_t::identity(dims, dims);
    const auto c  = -x0;
    const auto l  = make_random_vector<scalar_t>(dims);
    const auto u  = l.array() + 0.1;

    auto program  = make_quadratic(Q, c, make_greater(l), make_less(u));
    auto xbest    = x0.array().max(l.array()).min(u.array());
    auto expected = make_expected(xbest, program);

    return std::make_tuple(std::move(program), std::move(expected));
}
