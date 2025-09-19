#include <function/nonlinear/chung_reynolds.h>

using namespace nano;

function_chung_reynolds_t::function_chung_reynolds_t(const tensor_size_t dims)
    : function_t("chung-reynolds", dims)
{
    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_chung_reynolds_t::clone() const
{
    return std::make_unique<function_chung_reynolds_t>(*this);
}

scalar_t function_chung_reynolds_t::do_eval(eval_t eval) const
{
    const auto u = eval.m_x.dot(eval.m_x);

    if (eval.has_grad())
    {
        eval.m_gx = (4 * u) * eval.m_x;
    }

    if (eval.has_hess())
    {
        eval.m_Hx.matrix().noalias() = 8 * eval.m_x.vector() * eval.m_x.transpose();
        eval.m_Hx.diagonal().array() += 4 * u;
    }

    return u * u;
}

rfunction_t function_chung_reynolds_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_chung_reynolds_t>(dims);
}
