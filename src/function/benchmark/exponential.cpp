#include <nano/function/benchmark/exponential.h>

using namespace nano;

function_exponential_t::function_exponential_t(tensor_size_t dims)
    : benchmark_function_t("Exponential", dims)
{
    convex(true);
    smooth(true);
}

scalar_t function_exponential_t::vgrad(const vector_t& x, vector_t* gx) const
{
    const auto fx = std::exp(1 + x.dot(x) / scalar_t(size()));

    if (gx != nullptr)
    {
        gx->noalias() = (2 * fx / scalar_t(size())) * x;
    };

    return fx;
}

rfunction_t function_exponential_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_exponential_t>(dims);
}
