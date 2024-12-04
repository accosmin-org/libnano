#include <Eigen/Dense>
#include <function/program/cvx48d.h>
#include <nano/core/scat.h>

using namespace nano;

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
} // namespace

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
