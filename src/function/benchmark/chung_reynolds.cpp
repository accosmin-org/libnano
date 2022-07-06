#include <nano/function/benchmark/chung_reynolds.h>

using namespace nano;

function_chung_reynolds_t::function_chung_reynolds_t(tensor_size_t dims)
    : benchmark_function_t("Chung-Reynolds", dims)
{
    convex(true);
    smooth(true);
}

scalar_t function_chung_reynolds_t::vgrad(const vector_t& x, vector_t* gx) const
{
    const auto u = x.dot(x);

    if (gx != nullptr)
    {
        *gx = (4 * u) * x;
    }

    return u * u;
}

rfunction_t function_chung_reynolds_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_chung_reynolds_t>(dims);
}
