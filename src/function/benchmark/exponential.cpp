#include <nano/function/benchmark/exponential.h>

using namespace nano;

function_exponential_t::function_exponential_t(tensor_size_t dims)
    : benchmark_function_t("Exponential", dims)
{
    convex(true);
    smooth(true);
}

scalar_t function_exponential_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    const auto alpha = 1.0 / static_cast<scalar_t>(size());

    const auto fx = std::exp(1 + x.dot(x) * alpha);

    if (gx != nullptr)
    {
        gx->noalias() = (2 * fx * alpha) * x;
    };

    return fx;
}

rfunction_t function_exponential_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_exponential_t>(dims);
}
