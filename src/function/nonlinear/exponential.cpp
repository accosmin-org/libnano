#include <function/nonlinear/exponential.h>

using namespace nano;

function_exponential_t::function_exponential_t(const tensor_size_t dims)
    : function_t("exponential", dims)
{
    convex(convexity::yes);
    smooth(smoothness::yes);
    strong_convexity(2.0 / static_cast<scalar_t>(size()));
}

rfunction_t function_exponential_t::clone() const
{
    return std::make_unique<function_exponential_t>(*this);
}

scalar_t function_exponential_t::do_eval(eval_t eval) const
{
    const auto alpha = 1.0 / static_cast<scalar_t>(size());

    const auto fx = std::exp(1 + eval.m_x.dot(eval.m_x) * alpha);

    if (eval.has_grad())
    {
        eval.m_gx = (2 * fx * alpha) * eval.m_x.vector();
    }

    if (eval.has_hess())
    {
        eval.m_Hx = (4 * fx * alpha * alpha) * (eval.m_x.vector() * eval.m_x.transpose());
        eval.m_Hx.diagonal().array() += 2 * fx * alpha;
    }

    return fx;
}

rfunction_t function_exponential_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_exponential_t>(dims);
}
