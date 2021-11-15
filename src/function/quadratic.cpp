#include <nano/function/quadratic.h>

using namespace nano;

function_quadratic_t::function_quadratic_t(tensor_size_t dims) :
    benchmark_function_t("Quadratic", dims),
    m_a(vector_t::Random(dims))
{
    convex(true);
    smooth(true);

    // NB: generate random positive semi-definite matrix to keep the function convex
    matrix_t A = matrix_t::Random(dims, dims);
    m_A = matrix_t::Identity(dims, dims) + A * A.transpose();
}

scalar_t function_quadratic_t::vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->noalias() = m_a + m_A * x;
    }

    return x.dot(m_a + (m_A * x) / scalar_t(2));
}

rfunction_t function_quadratic_t::make(tensor_size_t dims) const
{
    return std::make_unique<function_quadratic_t>(dims);
}
