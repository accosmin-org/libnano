#include <function/nonlinear/geometric.h>
#include <nano/core/random.h>

using namespace nano;

namespace
{
auto make_samples(const tensor_size_t dims, const scalar_t sample_ratio)
{
    return static_cast<tensor_size_t>(std::max(sample_ratio * static_cast<scalar_t>(dims), 10.0));
}
} // namespace

function_geometric_optimization_t::function_geometric_optimization_t(const tensor_size_t dims, const uint64_t seed,
                                                                     const scalar_t sratio)
    : function_t("geometric-optimization", dims)
    , m_a(make_samples(dims, sratio))
    , m_A(make_samples(dims, sratio), dims)
{
    register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    register_parameter(parameter_t::make_scalar("function::geometric::sratio", 0.1, LE, sratio, LE, 1e+3));

    auto rng   = make_rng(seed);
    auto udist = make_udist<scalar_t>(-1.0, +1.0);

    m_a.full([&]() { return udist(rng); });
    m_A.full([&]() { return udist(rng) / static_cast<scalar_t>(dims); });

    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_geometric_optimization_t::clone() const
{
    return std::make_unique<function_geometric_optimization_t>(*this);
}

scalar_t function_geometric_optimization_t::do_vgrad(eval_t eval) const
{
    const auto a = m_a.vector();
    const auto A = m_A.matrix();

    if (eval.has_grad())
    {
        eval.m_gx = A.transpose() * (a + A * eval.m_x.vector()).array().exp().matrix();
    }

    if (eval.has_hess())
    {
        // TODO
    }

    return (a + A * eval.m_x.vector()).array().exp().sum();
}

string_t function_geometric_optimization_t::do_name() const
{
    const auto seed   = parameter("function::seed").value<uint64_t>();
    const auto sratio = parameter("function::geometric::sratio").value<scalar_t>();

    return scat(type_id(), "[sratio=", sratio, ",seed=", seed, "]");
}

rfunction_t function_geometric_optimization_t::make(const tensor_size_t dims) const
{
    const auto seed   = parameter("function::seed").value<uint64_t>();
    const auto sratio = parameter("function::geometric::sratio").value<scalar_t>();

    return std::make_unique<function_geometric_optimization_t>(dims, seed, sratio);
}
