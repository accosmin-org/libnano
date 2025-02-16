#include <function/benchmark/qing.h>

using namespace nano;

function_qing_t::function_qing_t(const tensor_size_t dims)
    : function_t("qing", dims)
    , m_bias(dims)
{
    m_bias.lin_spaced(scalar_t(1), scalar_t(dims));

    convex(convexity::no);
    smooth(smoothness::yes);
}

rfunction_t function_qing_t::clone() const
{
    return std::make_unique<function_qing_t>(*this);
}

scalar_t function_qing_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto xa = x.array();
    const auto ba = m_bias.array();

    if (gx.size() == x.size())
    {
        gx = 4 * (xa.square() - ba) * xa;
    }

    return (xa.square() - ba).square().sum();
}

rfunction_t function_qing_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_qing_t>(dims);
}
