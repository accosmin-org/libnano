#include <function/nonlinear/qing.h>

using namespace nano;

function_qing_t::function_qing_t(const tensor_size_t dims)
    : function_t("qing", dims)
    , m_bias(dims)
{
    m_bias.lin_spaced(scalar_t(1), scalar_t(dims));

    convex(convexity::no);
    smooth(smoothness::yes);
}

rfunction_t function_qing_t::clone() const
{
    return std::make_unique<function_qing_t>(*this);
}

scalar_t function_qing_t::do_eval(eval_t eval) const
{
    const auto x = eval.m_x.array();
    const auto b = m_bias.array();

    if (eval.has_grad())
    {
        eval.m_gx = 4 * (x.square() - b) * x;
    }

    if (eval.has_hess())
    {
        eval.m_Hx = (12 * x.square() - 4 * b).matrix().asDiagonal();
    }

    return (x.square() - b).square().sum();
}

rfunction_t function_qing_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_qing_t>(dims);
}
