#include <Eigen/Dense>
#include <function/program/cvx.h>

using namespace nano;
using namespace nano::program;

namespace
{
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

auto make_xbest_cvx48e_eq(const std::vector<std::pair<scalar_t, tensor_size_t>>& v, const tensor_size_t alpha)
{
    auto xbest = make_full_vector<scalar_t>(v.size(), 0.0);
    for (tensor_size_t i = 0; i < alpha; ++i)
    {
        const auto [value, index] = v[static_cast<size_t>(i)];
        xbest(index)              = 1.0;
    }
    return xbest;
}

auto make_xbest_cvx48e_ineq(const std::vector<std::pair<scalar_t, tensor_size_t>>& v, const tensor_size_t alpha)
{
    auto xbest = make_full_vector<scalar_t>(v.size(), 0.0);
    for (tensor_size_t i = 0, count = 0; i < v.size() && count < alpha; ++i)
    {
        const auto [value, index] = v[static_cast<size_t>(i)];
        if (value <= 0.0)
        {
            ++count;
            xbest(index) = 1.0;
        }
    }
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
    : linear_program_t(scat("cvx48b-[lambda=", lambda, "]"), dims)
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

linear_program_cvx48c_t::linear_program_cvx48c_t(const tensor_size_t dims)
    : linear_program_t("cvx48bc", dims)
{
    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto l = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto u = make_random_vector<scalar_t>(dims, +1.0, +3.0);

    reset(c);

    l <= (*this);
    (*this) <= u;

    xbest(l.array() * c.array().max(0.0).sign() - u.array() * c.array().min(0.0).sign());
}

rfunction_t linear_program_cvx48c_t::clone() const
{
    return std::make_unique<linear_program_cvx48c_t>(*this);
}

rfunction_t linear_program_cvx48c_t::make(const tensor_size_t dims, [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx48c_t>(dims);
}

linear_program_cvx48d_eq_t::linear_program_cvx48d_eq_t(const tensor_size_t dims)
    : linear_program_t("cvx48d-eq", dims)
{
    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto A = vector_t::constant(dims, 1.0);
    const auto b = 1.0;

    reset(c);

    (A * (*this)) == b;
    (*this) >= 0.0;

    xbest(make_xbest_cvx48d(c));
}

rfunction_t linear_program_cvx48d_eq_t::clone() const
{
    return std::make_unique<linear_program_cvx48d_eq_t>(*this);
}

rfunction_t linear_program_cvx48d_eq_t::make(const tensor_size_t                  dims,
                                             [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx48d_eq_t>(dims);
}

linear_program_cvx48d_ineq_t::linear_program_cvx48d_ineq_t(const tensor_size_t dims)
    : linear_program_t("cvx48d-ineq", dims)
{
    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto A = vector_t::constant(dims, 1.0);
    const auto N = -matrix_t::identity(dims, dims);
    const auto b = 1.0;
    const auto z = vector_t::constant(dims, 0.0);

    reset(c);

    (A * (*this)) <= b;
    (N * (*this)) <= z;
    (*this) >= 0.0;

    xbest(c.min() < 0.0 ? make_xbest_cvx48d(c) : make_full_vector<scalar_t>(dims, 0.0));
}

rfunction_t linear_program_cvx48d_ineq_t::clone() const
{
    return std::make_unique<linear_program_cvx48d_ineq_t>(*this);
}

rfunction_t linear_program_cvx48d_ineq_t::make(const tensor_size_t                  dims,
                                               [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx48d_ineq_t>(dims);
}

linear_program_cvx48e_eq_t::linear_program_cvx48e_eq_t(const tensor_size_t dims, const tensor_size_t alpha)
    : linear_program_t(scat("cvx48e-eq[alpha=", alpha, "]"), dims)
{
    assert(alpha >= 0);
    assert(alpha <= dims);

    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto a = make_full_vector<scalar_t>(dims, 1.0);
    const auto v = make_sorted_cvx48e(c);
    const auto h = static_cast<scalar_t>(alpha);

    reset(c);

    (a * (*this)) == h;
    (*this) >= 0.0;
    (*this) <= 1.0;

    xbest(make_xbest_48e_eq(v, alpha));
}

rfunction_t linear_program_cvx48e_eq_t::clone() const
{
    return std::make_unique<linear_program_cvx48e_eq_t>(*this);
}

rfunction_t linear_program_cvx48e_eq_t::make(const tensor_size_t                  dims,
                                             [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx48e_eq_t>(dims);
}

linear_program_cvx48e_ineq_t::linear_program_cvx48e_ineq_t(const tensor_size_t dims, const tensor_size_t alpha)
    : linear_program_t(scat("cvx48e-ineq[alpha=", alpha, "]"), dims)
{
    assert(alpha >= 0);
    assert(alpha <= dims);

    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto a = make_full_vector<scalar_t>(dims, 1.0);
    const auto v = make_sorted_cvx48e(c);
    const auto h = static_cast<scalar_t>(alpha);

    reset(c);

    (a * (*this)) <= h;
    (*this) >= 0.0;
    (*this) <= 1.0;

    xbest(make_xbest_48e_ineq(v, alpha));
}

rfunction_t linear_program_cvx48e_ineq_t::clone() const
{
    return std::make_unique<linear_program_cvx48e_ineq_t>(*this);
}

rfunction_t linear_program_cvx48e_ineq_t::make(const tensor_size_t                  dims,
                                               [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx48e_ineq_t>(dims);
}

linear_program_cvx48f_t::linear_program_cvx48f_t(const tensor_size_t dims, scalar_t alpha)
    : linear_program_t(scat("cvx48f[alpha=", alpha, "]"), dims)
{
    assert(alpha >= 0.0);
    assert(alpha <= 1.0);

    const auto d = make_random_vector<scalar_t>(dims, 1.0, +2.0);
    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto v = make_sorted_cvx48f(c, d);

    alpha = alpha * d.sum();

    reset(c);

    (d * (*this)) == alpha;
    (*this) >= 0.0;
    (*this) <= 1.0;

    auto accum = 0.0;
    auto xbest = make_full_vector<scalar_t>(dims, 0.0);
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
    this->xbest(xbest);
}

rfunction_t linear_program_cvx48f_t::clone() const
{
    return std::make_unique<linear_program_cvx48f_t>(*this);
}

rfunction_t linear_program_cvx48f_t::make(const tensor_size_t dims, [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx48f_t>(dims);
}

linear_program_cvx49_t::linear_program_cvx48f_t(const tensor_size_t dims)
    : linear_program_t("cvx49", dims)
{
    const auto c = make_random_vector<scalar_t>(dims, -1.0, -0.0);
    const auto A = matrix_t::identity(dims, dims);
    const auto b = make_random_vector<scalar_t>(dims, -1.0, +1.0);

    reset(c);

    (A * (*this)) <= b;

    xbest(b);
}

rfunction_t linear_program_cvx49_t::clone() const
{
    return std::make_unique<linear_program_cvx49_t>(*this);
}

rfunction_t linear_program_cvx49_t::make(const tensor_size_t dims, [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx49_t>(dims);
}

linear_program_cvx410_t::linear_program_cvx48f_t(const tensor_size_t dims, const bool feasible)
    : linear_program_t(scat("cvx410-[", feasible ? "feasible" : "unfeasible", "]"), dims)
{
    const auto D = make_random_matrix<scalar_t>(dims, dims);
    const auto A = D.transpose() * D + matrix_t::identity(dims, dims);
    const auto c = make_random_vector<scalar_t>(dims);

    if (feasible)
    {
        // the solution is feasible
        const auto x = make_random_vector<scalar_t>(dims, +1.0, +2.0);
        const auto b = A * x;

        reset(c);

        (A * (*this)) == b;
        (*this) >= 0.0;

        xbest(x);
    }
    else
    {
        // the solution is not feasible
        const auto x = make_random_vector<scalar_t>(dims, -2.0, -1.0);
        const auto b = A * x;

        reset(c);

        (A * (*this)) == b;
        (*this) >= 0.0;

        xbest(x);
        expected.status(solver_status::unfeasible);
    }
}

rfunction_t linear_program_cvx410_t::clone() const
{
    return std::make_unique<linear_program_cvx410_t>(*this);
}

rfunction_t linear_program_cvx410_t::make(const tensor_size_t dims, [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx410_t>(dims);
}
