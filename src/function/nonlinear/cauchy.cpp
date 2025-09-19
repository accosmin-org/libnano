#include <function/nonlinear/cauchy.h>

using namespace nano;

function_cauchy_t::function_cauchy_t(const tensor_size_t dims)
    : function_t("cauchy", dims)
{
    convex(convexity::no);
    smooth(smoothness::yes);
}

rfunction_t function_cauchy_t::clone() const
{
    return std::make_unique<function_cauchy_t>(*this);
}

scalar_t function_cauchy_t::do_eval(eval_t eval) const
{
    const auto xTx = eval.m_x.dot(eval.m_x);

    if (eval.has_grad())
    {
        eval.m_gx = 2 * eval.m_x / (1 + xTx);
    }

    if (eval.has_hess())
    {
        eval.m_Hx.matrix().noalias() = -4 * (eval.m_x.vector() / (1 + xTx)) * (eval.m_x.transpose() / (1 + xTx));
        eval.m_Hx.diagonal().array() += 2 / (1 + xTx);
    }

    return std::log1p(xTx);
}

rfunction_t function_cauchy_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_cauchy_t>(dims);
}
