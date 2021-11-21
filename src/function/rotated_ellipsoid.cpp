#include <nano/core/numeric.h>
#include <nano/function/rotated_ellipsoid.h>

using namespace nano;

function_rotated_ellipsoid_t::function_rotated_ellipsoid_t(tensor_size_t dims) :
    benchmark_function_t("Rotated Hyper-Ellipsoid", dims)
{
    convex(true);
    smooth(true);
}

scalar_t function_rotated_ellipsoid_t::vgrad(const vector_t& x, vector_t* gx) const
{
    scalar_t fx = 0, fi = 0;
    for (tensor_size_t i = 0; i < size(); i ++)
    {
        fi += x(i);
        fx += nano::square(fi);
        if (gx != nullptr)
        {
            (*gx)(i) = 2 * fi;
        }
    }

    if (gx != nullptr)
    {
        for (auto i = size() - 2; i >= 0; i --)
        {
            (*gx)(i) += (*gx)(i + 1);
        }
    }

    return fx;
}

rfunction_t function_rotated_ellipsoid_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_rotated_ellipsoid_t>(dims);
}
