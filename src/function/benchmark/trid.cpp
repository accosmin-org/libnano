#include <nano/function/benchmark/trid.h>

using namespace nano;

function_trid_t::function_trid_t(tensor_size_t dims)
    : function_t("trid", dims)
{
    convex(true);
    smooth(true);
}

rfunction_t function_trid_t::clone() const
{
    return std::make_unique<function_trid_t>(*this);
}

scalar_t function_trid_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        *gx = 2 * (x.array() - 1);
        gx->segment(1, size() - 1) -= x.segment(0, size() - 1);
        gx->segment(0, size() - 1) -= x.segment(1, size() - 1);
    }

    return (x.array() - 1).square().sum() - (x.segment(0, size() - 1).array() * x.segment(1, size() - 1).array()).sum();
}

rfunction_t function_trid_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_trid_t>(dims);
}
