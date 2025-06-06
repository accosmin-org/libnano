#include <function/lp/cvx48e.h>
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

auto make_alpha(const tensor_size_t dims, const scalar_t alpha, const tensor_size_t min_alpha)
{
    return std::max(static_cast<tensor_size_t>(alpha * static_cast<scalar_t>(dims)), min_alpha);
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

linear_program_cvx48e_eq_t::linear_program_cvx48e_eq_t(const tensor_size_t dims, const uint64_t seed,
                                                       const scalar_t alpha)
    : linear_program_t("cvx48e-eq", vector_t::zero(dims))
{
    parameter("function::seed") = seed;
    register_parameter(parameter_t::make_scalar("function::cvx48e-eq::alpha", 0.0, LE, alpha, LE, 1.0));

    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0, seed);
    const auto a = make_full_vector<scalar_t>(dims, 1.0);
    const auto v = make_sorted_cvx48e(c);
    const auto h = static_cast<scalar_t>(make_alpha(dims, alpha, 0));

    this->c() = c;
    optimum(make_xbest_cvx48e_eq(v, make_alpha(dims, alpha, 0)));

    critical((a * variable()) == h);
    critical(variable() >= 0.0);
    critical(variable() <= 1.0);
}

rfunction_t linear_program_cvx48e_eq_t::clone() const
{
    return std::make_unique<linear_program_cvx48e_eq_t>(*this);
}

string_t linear_program_cvx48e_eq_t::do_name() const
{
    const auto seed  = parameter("function::seed").value<uint64_t>();
    const auto alpha = parameter("function::cvx48e-eq::alpha").value<scalar_t>();

    return scat(type_id(), "[alpha=", alpha, ",seed=", seed, "]");
}

rfunction_t linear_program_cvx48e_eq_t::make(const tensor_size_t dims) const
{
    const auto seed  = parameter("function::seed").value<uint64_t>();
    const auto alpha = parameter("function::cvx48e-eq::alpha").value<scalar_t>();

    return std::make_unique<linear_program_cvx48e_eq_t>(dims, seed, alpha);
}

linear_program_cvx48e_ineq_t::linear_program_cvx48e_ineq_t(const tensor_size_t dims, const uint64_t seed,
                                                           const scalar_t alpha)
    : linear_program_t("cvx48e-ineq", vector_t::zero(dims))
{
    parameter("function::seed") = seed;
    register_parameter(parameter_t::make_scalar("function::cvx48e-ineq::alpha", 0.0, LT, alpha, LE, 1.0));

    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0, seed);
    const auto a = make_full_vector<scalar_t>(dims, 1.0);
    const auto v = make_sorted_cvx48e(c);
    const auto h = static_cast<scalar_t>(make_alpha(dims, alpha, 1));

    this->c() = c;
    optimum(make_xbest_cvx48e_ineq(v, make_alpha(dims, alpha, 1)));

    critical((a * variable()) <= h);
    critical(variable() >= 0.0);
    critical(variable() <= 1.0);
}

rfunction_t linear_program_cvx48e_ineq_t::clone() const
{
    return std::make_unique<linear_program_cvx48e_ineq_t>(*this);
}

string_t linear_program_cvx48e_ineq_t::do_name() const
{
    const auto seed  = parameter("function::seed").value<uint64_t>();
    const auto alpha = parameter("function::cvx48e-ineq::alpha").value<scalar_t>();

    return scat(type_id(), "[alpha=", alpha, ",seed=", seed, "]");
}

rfunction_t linear_program_cvx48e_ineq_t::make(const tensor_size_t dims) const
{
    const auto seed  = parameter("function::seed").value<uint64_t>();
    const auto alpha = parameter("function::cvx48e-ineq::alpha").value<scalar_t>();

    return std::make_unique<linear_program_cvx48e_ineq_t>(dims, seed, alpha);
}
