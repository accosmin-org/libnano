#include <nano/function/benchmark/quadratic.h>

using namespace nano;

function_quadratic_t::function_quadratic_t(tensor_size_t dims)
    : benchmark_function_t("Quadratic", dims)
    , m_a(make_random_vector<scalar_t>(dims, -1.0, +1.0, seed_t{42}))
{
    convex(true);
    smooth(true);

    // NB: generate random positive semi-definite matrix to keep the function convex
    matrix_t A = make_random_matrix<scalar_t>(dims, dims, -1.0, +1.0, seed_t{42});
    m_A        = matrix_t::Identity(dims, dims) + A * A.transpose();
}

scalar_t function_quadratic_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->noalias() = m_a + m_A * x;
    }

    return x.dot(m_a + (m_A * x) / scalar_t(2));
}

rfunction_t function_quadratic_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_quadratic_t>(dims);
}
