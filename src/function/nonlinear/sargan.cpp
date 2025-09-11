#include <function/nonlinear/sargan.h>
#include <nano/core/numeric.h>

using namespace nano;

function_sargan_t::function_sargan_t(const tensor_size_t dims)
    : function_t("sargan", dims)
{
    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_sargan_t::clone() const
{
    return std::make_unique<function_sargan_t>(*this);
}

scalar_t function_sargan_t::do_eval(eval_t eval) const
{
    const auto xsum = eval.m_x.sum();

    if (eval.has_grad())
    {
        eval.m_gx = (1.2 + 0.8 * xsum) * eval.m_x;
    }

    if (eval.has_hess())
    {
        eval.m_Hx.matrix().colwise() = 0.8 * eval.m_x.vector();
        eval.m_Hx.diagonal().array() += 1.2 + 0.8 * xsum;
    }

    return 0.6 * eval.m_x.dot(eval.m_x) + 0.4 * nano::square(xsum);
}

rfunction_t function_sargan_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_sargan_t>(dims);
}
