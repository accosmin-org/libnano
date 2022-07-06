#include <nano/function/benchmark/qing.h>

using namespace nano;

function_qing_t::function_qing_t(tensor_size_t dims)
    : benchmark_function_t("Qing", dims)
    , m_bias(vector_t::LinSpaced(dims, scalar_t(1), scalar_t(dims)))
{
    convex(false);
    smooth(true);
}

scalar_t function_qing_t::vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        *gx = 4 * (x.array().square() - m_bias.array()) * x.array();
    }

    return (x.array().square() - m_bias.array()).square().sum();
}

rfunction_t function_qing_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_qing_t>(dims);
}
