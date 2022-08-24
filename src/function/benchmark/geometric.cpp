#include <nano/core/random.h>
#include <nano/function/benchmark/geometric.h>

using namespace nano;

function_geometric_optimization_t::function_geometric_optimization_t(tensor_size_t dims, tensor_size_t summands)
    : benchmark_function_t("Geometric Optimization", dims)
    , m_a(make_random_vector<scalar_t>(summands, -1.0, +1.0, seed_t{42}))
    , m_A(make_random_matrix<scalar_t>(summands, dims, -1.0 / static_cast<scalar_t>(dims),
                                       +1.0 / static_cast<scalar_t>(dims), seed_t{42}))
{
    assert(summands > 0);
    convex(true);
    smooth(true);
}

scalar_t function_geometric_optimization_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->noalias() = m_A.transpose() * (m_a + m_A * x).array().exp().matrix();
    }

    return (m_a + m_A * x).array().exp().sum();
}

rfunction_t function_geometric_optimization_t::make(tensor_size_t dims, tensor_size_t summands) const
{
    return std::make_unique<function_geometric_optimization_t>(dims, summands);
}
