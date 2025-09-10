#include <function/nonlinear/cauchy.h>

using namespace nano;

function_cauchy_t::function_cauchy_t(const tensor_size_t dims)
    : function_t("cauchy", dims)
{
    convex(convexity::no);
    smooth(smoothness::yes);
}

rfunction_t function_cauchy_t::clone() const
{
    return std::make_unique<function_cauchy_t>(*this);
}

scalar_t function_cauchy_t::do_eval(eval_t eval) const
{
    if (gx.size() == x.size())
    {
        gx = 2 * x / (1 + x.dot(x));
    }

    return std::log1p(x.dot(x));
}

rfunction_t function_cauchy_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_cauchy_t>(dims);
}
