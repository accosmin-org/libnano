#include <function/benchmark/trid.h>

using namespace nano;

function_trid_t::function_trid_t(const tensor_size_t dims)
    : function_t("trid", dims)
{
    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_trid_t::clone() const
{
    return std::make_unique<function_trid_t>(*this);
}

scalar_t function_trid_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    if (gx.size() == x.size())
    {
        gx = 2 * (x.array() - 1);
        gx.segment(1, size() - 1) -= x.segment(0, size() - 1);
        gx.segment(0, size() - 1) -= x.segment(1, size() - 1);
    }

    return (x.array() - 1).square().sum() - (x.segment(0, size() - 1).array() * x.segment(1, size() - 1).array()).sum();
}

rfunction_t function_trid_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_trid_t>(dims);
}
