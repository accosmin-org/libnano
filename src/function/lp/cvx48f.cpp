#include <function/lp/cvx48f.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>

using namespace nano;

namespace
{
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

auto make_xbest_cvx48f(const vector_t& d, const std::vector<std::pair<scalar_t, tensor_size_t>>& v,
                       const scalar_t alpha)
{
    const auto dims = static_cast<tensor_size_t>(v.size());

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
    return xbest;
}
} // namespace

linear_program_cvx48f_t::linear_program_cvx48f_t(const tensor_size_t dims, const uint64_t seed, scalar_t alpha)
    : linear_program_t("cvx48f", vector_t::zero(dims))
{
    register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    register_parameter(parameter_t::make_scalar("function::cvx48f::alpha", 0.0, LE, alpha, LE, 1.0));

    auto rng   = make_rng(seed);
    auto udist = make_udist<scalar_t>(-1.0, +1.0);

    const auto d = make_full_vector<scalar_t>(dims, [&]() { return udist(rng) * 0.5 + 1.5; });
    const auto c = make_full_vector<scalar_t>(dims, [&]() { return udist(rng); });
    const auto v = make_sorted_cvx48f(c, d);

    alpha = alpha * d.sum();

    this->c() = c;
    optimum(make_xbest_cvx48f(d, v, alpha));

    critical((d * variable()) == alpha);
    critical(variable() >= 0.0);
    critical(variable() <= 1.0);
}

rfunction_t linear_program_cvx48f_t::clone() const
{
    return std::make_unique<linear_program_cvx48f_t>(*this);
}

string_t linear_program_cvx48f_t::do_name() const
{
    const auto seed  = parameter("function::seed").value<uint64_t>();
    const auto alpha = parameter("function::cvx48f::alpha").value<scalar_t>();

    return scat(type_id(), "[alpha=", alpha, ",seed=", seed, "]");
}

rfunction_t linear_program_cvx48f_t::make(const tensor_size_t dims) const
{
    const auto seed  = parameter("function::seed").value<uint64_t>();
    const auto alpha = parameter("function::cvx48f::alpha").value<scalar_t>();

    return std::make_unique<linear_program_cvx48f_t>(dims, seed, alpha);
}
