#include <function/benchmark/axis_ellipsoid.h>

using namespace nano;

function_axis_ellipsoid_t::function_axis_ellipsoid_t(tensor_size_t dims)
    : function_t("axis-ellipsoid", dims)
    , m_bias(dims)
{
    m_bias.lin_spaced(scalar_t(1), scalar_t(dims));

    convex(convexity::yes);
    smooth(smoothness::yes);
    strong_convexity(2.0);
}

rfunction_t function_axis_ellipsoid_t::clone() const
{
    return std::make_unique<function_axis_ellipsoid_t>(*this);
}

scalar_t function_axis_ellipsoid_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    if (gx.size() == x.size())
    {
        gx = 2 * x.array() * m_bias.array();
    }

    return (x.array().square() * m_bias.array()).sum();
}

rfunction_t function_axis_ellipsoid_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_axis_ellipsoid_t>(dims);
}
