#include <nano/function/sphere.h>

using namespace nano;

function_sphere_t::function_sphere_t(tensor_size_t dims) :
    benchmark_function_t("Sphere", dims)
{
    convex(true);
    smooth(true);
}

scalar_t function_sphere_t::vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->noalias() = 2 * x;
    }

    return x.dot(x);
}

rfunction_t function_sphere_t::make(tensor_size_t dims) const
{
    return std::make_unique<function_sphere_t>(dims);
}
