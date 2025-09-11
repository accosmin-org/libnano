#include <function/nonlinear/rotated_ellipsoid.h>
#include <nano/core/numeric.h>

using namespace nano;

function_rotated_ellipsoid_t::function_rotated_ellipsoid_t(const tensor_size_t dims)
    : function_t("rotated-ellipsoid", dims)
{
    m_bias.lin_spaced(scalar_t(dims + 1), scalar_t(1));

    convex(convexity::yes);
    smooth(smoothness::yes);
    strong_convexity(2.0);
}

rfunction_t function_rotated_ellipsoid_t::clone() const
{
    return std::make_unique<function_rotated_ellipsoid_t>(*this);
}

scalar_t function_rotated_ellipsoid_t::do_eval(eval_t eval) const
{
    if (eval.has_grad())
    {
        eval.m_gx = 2 * eval.m_x.array() * m_bias.array();
    }

    if (eval.has_hess())
    {
        eval.m_Hx = (2 * m_bias.array()).matrix().asDiagonal();
    }

    return (eval.m_x.array().square() * m_bias.array()).sum();
}

rfunction_t function_rotated_ellipsoid_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_rotated_ellipsoid_t>(dims);
}
