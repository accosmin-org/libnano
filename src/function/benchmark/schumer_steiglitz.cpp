#include <nano/function/benchmark/schumer_steiglitz.h>

using namespace nano;

function_schumer_steiglitz_t::function_schumer_steiglitz_t(tensor_size_t dims)
    : benchmark_function_t("Schumer-Steiglitz", dims)
{
    convex(true);
    smooth(true);
}

scalar_t function_schumer_steiglitz_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        *gx = 4 * x.array().cube();
    }

    return x.array().square().square().sum();
}

rfunction_t function_schumer_steiglitz_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_schumer_steiglitz_t>(dims);
}
