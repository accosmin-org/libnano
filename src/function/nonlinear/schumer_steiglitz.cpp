#include <function/nonlinear/schumer_steiglitz.h>

using namespace nano;

function_schumer_steiglitz_t::function_schumer_steiglitz_t(const tensor_size_t dims)
    : function_t("schumer-steiglitz", dims)
{
    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_schumer_steiglitz_t::clone() const
{
    return std::make_unique<function_schumer_steiglitz_t>(*this);
}

scalar_t function_schumer_steiglitz_t::do_eval(eval_t eval) const
{
    if (eval.has_grad())
    {
        eval.m_gx = 4 * eval.m_x.array().cube();
    }

    if (eval.has_hess())
    {
        eval.m_hx = (12 * eval.m_x.array().square()).matrix().asDiagonal();
    }

    return eval.m_x.array().square().square().sum();
}

rfunction_t function_schumer_steiglitz_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_schumer_steiglitz_t>(dims);
}
