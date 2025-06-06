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
    , m_a(make_random_vector<scalar_t>(make_samples(dims, sratio), -1.0, +1.0, seed))
    , m_A(make_random_matrix<scalar_t>(make_samples(dims, sratio), dims, -1.0 / static_cast<scalar_t>(dims),
                                       +1.0 / static_cast<scalar_t>(dims), seed))
{
    parameter("function::seed") = seed;
    register_parameter(parameter_t::make_scalar("function::geometric::sratio", 0.1, LE, sratio, LE, 1e+3));

    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_geometric_optimization_t::clone() const
{
    return std::make_unique<function_geometric_optimization_t>(*this);
}

scalar_t function_geometric_optimization_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto a = m_a.vector();
    const auto A = m_A.matrix();

    if (gx.size() == x.size())
    {
        gx = A.transpose() * (a + A * x.vector()).array().exp().matrix();
    }

    return (a + A * x.vector()).array().exp().sum();
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
