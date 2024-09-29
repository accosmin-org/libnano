#include <function/benchmark/sphere.h>

using namespace nano;

function_sphere_t::function_sphere_t(const tensor_size_t dims)
    : function_t("sphere", dims)
{
    convex(convexity::yes);
    smooth(smoothness::yes);
    strong_convexity(2.0);
}

rfunction_t function_sphere_t::clone() const
{
    return std::make_unique<function_sphere_t>(*this);
}

scalar_t function_sphere_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    if (gx.size() == x.size())
    {
        gx = 2 * x;
    }

    return x.dot(x);
}

rfunction_t function_sphere_t::make(const tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_sphere_t>(dims);
}
