#include <nano/function/benchmark/sphere.h>

using namespace nano;

function_sphere_t::function_sphere_t(tensor_size_t dims)
    : function_t("sphere", dims)
{
    convex(true);
    smooth(true);
    strong_convexity(2.0);
}

rfunction_t function_sphere_t::clone() const
{
    return std::make_unique<function_sphere_t>(*this);
}

scalar_t function_sphere_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->noalias() = 2 * x;
    }

    return x.dot(x);
}

rfunction_t function_sphere_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_sphere_t>(dims);
}
