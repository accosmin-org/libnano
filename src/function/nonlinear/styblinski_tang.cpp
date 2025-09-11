#include <function/nonlinear/styblinski_tang.h>

using namespace nano;

function_styblinski_tang_t::function_styblinski_tang_t(const tensor_size_t dims)
    : function_t("styblinski-tang", dims)
{
    convex(convexity::no);
    smooth(smoothness::yes);
}

rfunction_t function_styblinski_tang_t::clone() const
{
    return std::make_unique<function_styblinski_tang_t>(*this);
}

scalar_t function_styblinski_tang_t::do_eval(eval_t eval) const
{
    if (eval.has_grad())
    {
        eval.m_gx = 4 * eval.m_x.array().cube() - 32 * eval.m_x.array() + 5;
    }

    if (eval.has_hess())
    {
        eval.m_Hx = (12 * eval.m_x.array().square() - 32).matrix().asDiagonal();
    }

    return (eval.m_x.array().square().square() - 16 * eval.m_x.array().square() + 5 * eval.m_x.array()).sum();
}

rfunction_t function_styblinski_tang_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_styblinski_tang_t>(dims);
}
