#include <function/benchmark/geometric.h>
#include <nano/core/random.h>

using namespace nano;

function_geometric_optimization_t::function_geometric_optimization_t(tensor_size_t dims, tensor_size_t summands)
    : function_t("geometric-optimization", dims)
    , m_a(make_random_vector<scalar_t>(summands, -1.0, +1.0, seed_t{42}))
    , m_A(make_random_matrix<scalar_t>(summands, dims, -1.0 / static_cast<scalar_t>(dims),
                                       +1.0 / static_cast<scalar_t>(dims), seed_t{42}))
{
    assert(summands > 0);
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

rfunction_t function_geometric_optimization_t::make(tensor_size_t dims, tensor_size_t summands) const
{
    return std::make_unique<function_geometric_optimization_t>(dims, summands);
}
