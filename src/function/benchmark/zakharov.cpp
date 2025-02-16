#include <function/benchmark/zakharov.h>
#include <nano/core/numeric.h>

using namespace nano;

function_zakharov_t::function_zakharov_t(const tensor_size_t dims)
    : function_t("zakharov", dims)
    , m_bias(dims)
{
    m_bias.lin_spaced(scalar_t(0.5), scalar_t(dims) / scalar_t(2));

    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_zakharov_t::clone() const
{
    return std::make_unique<function_zakharov_t>(*this);
}

scalar_t function_zakharov_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto u = x.dot(x);
    const auto v = x.dot(m_bias);

    if (gx.size() == x.size())
    {
        gx = 2 * x + (2 * v + 4 * nano::cube(v)) * m_bias;
    }

    return u + nano::square(v) + nano::quartic(v);
}

rfunction_t function_zakharov_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_zakharov_t>(dims);
}
