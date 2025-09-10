#include <function/nonlinear/rotated_ellipsoid.h>
#include <nano/core/numeric.h>

using namespace nano;

function_rotated_ellipsoid_t::function_rotated_ellipsoid_t(const tensor_size_t dims)
    : function_t("rotated-ellipsoid", dims)
{
    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_rotated_ellipsoid_t::clone() const
{
    return std::make_unique<function_rotated_ellipsoid_t>(*this);
}

scalar_t function_rotated_ellipsoid_t::do_eval(eval_t eval) const
{
    scalar_t fi = 0;
    scalar_t fx = 0;
    for (tensor_size_t i = 0; i < size(); i++)
    {
        fi += x(i);
        fx += nano::square(fi);
        if (gx.size() == x.size())
        {
            gx(i) = 2 * fi;
        }
    }

    if (gx.size() == x.size())
    {
        for (auto i = size() - 2; i >= 0; i--)
        {
            gx(i) += gx(i + 1);
        }
    }

    return fx;
}

rfunction_t function_rotated_ellipsoid_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_rotated_ellipsoid_t>(dims);
}
