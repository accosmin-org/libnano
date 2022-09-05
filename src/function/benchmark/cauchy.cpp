#include <nano/function/benchmark/cauchy.h>

using namespace nano;

function_cauchy_t::function_cauchy_t(tensor_size_t dims)
    : function_t("cauchy", dims)
{
    convex(false);
    smooth(true);
}

rfunction_t function_cauchy_t::clone() const
{
    return std::make_unique<function_cauchy_t>(*this);
}

scalar_t function_cauchy_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        *gx = 2 * x / (1 + x.dot(x));
    }

    return std::log1p(x.dot(x));
}

rfunction_t function_cauchy_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_cauchy_t>(dims);
}
