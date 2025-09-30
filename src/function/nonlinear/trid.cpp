#include <function/nonlinear/trid.h>

using namespace nano;

function_trid_t::function_trid_t(const tensor_size_t dims)
    : function_t("trid", dims)
{
    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_trid_t::clone() const
{
    return std::make_unique<function_trid_t>(*this);
}

scalar_t function_trid_t::do_eval(eval_t eval) const
{
    const auto x = eval.m_x.vector();

    if (eval.has_grad())
    {
        eval.m_gx = 2 * (x.array() - 1);
        eval.m_gx.segment(1, size() - 1) -= x.segment(0, size() - 1);
        eval.m_gx.segment(0, size() - 1) -= x.segment(1, size() - 1);
    }

    if (eval.has_hess())
    {
        eval.m_hx = 2 * matrix_t::identity(size(), size());
        for (tensor_size_t i = 0, size = this->size(); i + 1 < size; ++i)
        {
            eval.m_hx(i, i + 1) -= 1.0;
        }
        for (tensor_size_t i = 1, size = this->size(); i < size; ++i)
        {
            eval.m_hx(i, i - 1) -= 1.0;
        }
    }

    return (x.array() - 1).square().sum() - (x.segment(0, size() - 1).array() * x.segment(1, size() - 1).array()).sum();
}

rfunction_t function_trid_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_trid_t>(dims);
}
