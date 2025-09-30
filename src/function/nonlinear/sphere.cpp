#include <function/nonlinear/sphere.h>

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

scalar_t function_sphere_t::do_eval(eval_t eval) const
{
    if (eval.has_grad())
    {
        eval.m_gx = eval.m_x;
    }

    if (eval.has_hess())
    {
        eval.m_hx = matrix_t::identity(size(), size());
    }

    return 0.5 * eval.m_x.dot(eval.m_x);
}

rfunction_t function_sphere_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_sphere_t>(dims);
}
