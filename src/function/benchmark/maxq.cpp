#include <function/benchmark/maxq.h>

using namespace nano;

function_maxq_t::function_maxq_t(const tensor_size_t dims)
    : function_t("maxq", dims)
{
    convex(convexity::yes);
    smooth(smoothness::no);
    strong_convexity(0.0);
}

rfunction_t function_maxq_t::clone() const
{
    return std::make_unique<function_maxq_t>(*this);
}

scalar_t function_maxq_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    auto       idx = tensor_size_t{0};
    const auto fx  = x.array().square().maxCoeff(&idx);

    if (gx.size() == x.size())
    {
        gx.array() = 0.0;
        gx(idx)    = 2.0 * x(idx);
    }

    return fx;
}

rfunction_t function_maxq_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_maxq_t>(dims);
}
