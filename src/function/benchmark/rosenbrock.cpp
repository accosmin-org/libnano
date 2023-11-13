#include <nano/core/numeric.h>
#include <nano/function/benchmark/rosenbrock.h>

using namespace nano;

function_rosenbrock_t::function_rosenbrock_t(tensor_size_t dims)
    : function_t("rosenbrock", std::max(dims, tensor_size_t(2)))
{
    convex(convexity::no);
    smooth(smoothness::yes);
}

rfunction_t function_rosenbrock_t::clone() const
{
    return std::make_unique<function_rosenbrock_t>(*this);
}

scalar_t function_rosenbrock_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto ct = scalar_t(100);

    scalar_t fx = 0;
    for (tensor_size_t i = 0; i + 1 < size(); ++i)
    {
        fx += ct * nano::square(x(i + 1) - x(i) * x(i)) + nano::square(x(i) - 1);
    }

    if (gx.size() == x.size())
    {
        gx.full(0.0);
        for (tensor_size_t i = 0; i + 1 < size(); ++i)
        {
            gx(i) += 2 * (x(i) - 1);
            gx(i) += ct * 2 * (x(i + 1) - x(i) * x(i)) * (-2 * x(i));
            gx(i + 1) += ct * 2 * (x(i + 1) - x(i) * x(i));
        }
    }

    return fx;
}

rfunction_t function_rosenbrock_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_rosenbrock_t>(dims);
}
