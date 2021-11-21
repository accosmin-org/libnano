#include <nano/function/styblinski_tang.h>

using namespace nano;

function_styblinski_tang_t::function_styblinski_tang_t(tensor_size_t dims) :
    benchmark_function_t("Styblinski-Tang", dims)
{
    convex(false);
    smooth(true);
}

scalar_t function_styblinski_tang_t::vgrad(const vector_t& x, vector_t* gx, vgrad_config_t) const
{
    if (gx != nullptr)
    {
        *gx = 4 * x.array().cube() - 32 * x.array() + 5;
    }

    return (x.array().square().square() - 16 * x.array().square() + 5 * x.array()).sum();
}

rfunction_t function_styblinski_tang_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_styblinski_tang_t>(dims);
}
