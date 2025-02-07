#include <function/program/cvx48e.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>

using namespace nano;

namespace
{
auto make_xbest_cvx48e_eq(const std::vector<std::pair<scalar_t, tensor_size_t>>& v, const tensor_size_t alpha)
{
    const auto dims = static_cast<tensor_size_t>(v.size());

    auto xbest = make_full_vector<scalar_t>(dims, 0.0);
    for (tensor_size_t i = 0; i < alpha; ++i)
    {
        const auto [value, index] = v[static_cast<size_t>(i)];
        xbest(index)              = 1.0;
    }
    return xbest;
}

auto make_xbest_cvx48e_ineq(const std::vector<std::pair<scalar_t, tensor_size_t>>& v, const tensor_size_t alpha)
{
    const auto dims = static_cast<tensor_size_t>(v.size());

    auto xbest = make_full_vector<scalar_t>(dims, 0.0);
    for (tensor_size_t i = 0, count = 0; i < dims && count < alpha; ++i)
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
} // namespace

linear_program_cvx48e_eq_t::linear_program_cvx48e_eq_t(const tensor_size_t dims, const scalar_t alpha_dims_ratio)
    : linear_program_t(scat("cvx48e-eq[alpha=", alpha_dims_ratio, "]"), vector_t::zero(dims))
{
    register_parameter(parameter_t::make_scalar("cvx48e-eq::alpha_ratio", 0.0, LE, 0.0, LE, 1.0));

    parameter("cvx48e-eq::alpha_ratio") = alpha_dims_ratio;

    const auto alpha = static_cast<tensor_size_t>(alpha_dims_ratio * static_cast<scalar_t>(dims));
    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto a = make_full_vector<scalar_t>(dims, 1.0);
    const auto v = make_sorted_cvx48e(c);
    const auto h = static_cast<scalar_t>(alpha);

    reset(c);
    optimum(make_xbest_cvx48e_eq(v, alpha));

    critical((a * variable()) == h);
    critical(variable() >= 0.0);
    critical(variable() <= 1.0);
}

rfunction_t linear_program_cvx48e_eq_t::clone() const
{
    return std::make_unique<linear_program_cvx48e_eq_t>(*this);
}

rfunction_t linear_program_cvx48e_eq_t::make(const tensor_size_t dims) const
{
    const auto alpha_dims_ratio = parameter("cvx48e-eq::alpha_ratio").value<scalar_t>();

    return std::make_unique<linear_program_cvx48e_eq_t>(dims, alpha_dims_ratio);
}

linear_program_cvx48e_ineq_t::linear_program_cvx48e_ineq_t(const tensor_size_t dims, const scalar_t alpha_dims_ratio)
    : linear_program_t(scat("cvx48e-ineq[alpha=", alpha_dims_ratio, "]"), vector_t::zero(dims))
{
    register_parameter(parameter_t::make_scalar("cvx48e-ineq::alpha_ratio", 0.0, LE, 0.0, LE, 1.0));

    parameter("cvx48e-ineq::alpha_ratio") = alpha_dims_ratio;

    const auto alpha = static_cast<tensor_size_t>(alpha_dims_ratio * static_cast<scalar_t>(dims));
    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto a = make_full_vector<scalar_t>(dims, 1.0);
    const auto v = make_sorted_cvx48e(c);
    const auto h = static_cast<scalar_t>(alpha);

    reset(c);
    optimum(make_xbest_cvx48e_ineq(v, alpha));

    critical((a * variable()) <= h);
    critical(variable() >= 0.0);
    critical(variable() <= 1.0);
}

rfunction_t linear_program_cvx48e_ineq_t::clone() const
{
    return std::make_unique<linear_program_cvx48e_ineq_t>(*this);
}

rfunction_t linear_program_cvx48e_ineq_t::make(const tensor_size_t dims) const
{
    const auto alpha_dims_ratio = parameter("cvx48e-ineq::alpha_ratio").value<scalar_t>();

    return std::make_unique<linear_program_cvx48e_ineq_t>(dims, alpha_dims_ratio);
}
