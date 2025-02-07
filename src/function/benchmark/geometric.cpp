#include <function/benchmark/geometric.h>
#include <nano/core/random.h>

using namespace nano;

namespace
{
auto make_samples(const tensor_size_t dims, const scalar_t sample_ratio)
{
    return static_cast<tensor_size_t>(std::max(sample_ratio * static_cast<scalar_t>(dims), 10.0));
}
} // namespace

function_geometric_optimization_t::function_geometric_optimization_t(const tensor_size_t dims,
                                                                     const scalar_t      sample_ratio)
    : function_t("geometric-optimization", dims)
    , m_a(make_random_vector<scalar_t>(make_samples(dims, sample_ratio), -1.0, +1.0, seed_t{42}))
    , m_A(make_random_matrix<scalar_t>(make_samples(dims, sample_ratio), dims, -1.0 / static_cast<scalar_t>(dims),
                                       +1.0 / static_cast<scalar_t>(dims), seed_t{42}))
{
    register_parameter(parameter_t::make_scalar("geometric-optimization::sratio", 0.1, LE, 10.0, LE, 1e+3));

    parameter("geometric-optimization::sratio") = sample_ratio;

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

rfunction_t function_geometric_optimization_t::make(const tensor_size_t dims) const
{
    const auto sratio = parameter("enet::sratio").value<scalar_t>();

    return std::make_unique<function_geometric_optimization_t>(dims, sratio);
}
