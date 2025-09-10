#include <function/nonlinear/dixon_price.h>
#include <nano/core/numeric.h>

using namespace nano;

function_dixon_price_t::function_dixon_price_t(const tensor_size_t dims)
    : function_t("dixon-price", dims)
    , m_bias(dims)
{
    m_bias.lin_spaced(scalar_t(1), scalar_t(dims));

    convex(convexity::no);
    smooth(smoothness::yes);
}

rfunction_t function_dixon_price_t::clone() const
{
    return std::make_unique<function_dixon_price_t>(*this);
}

scalar_t function_dixon_price_t::do_eval(eval_t eval) const
{
    const auto xsegm0 = eval.m_x.segment(0, size() - 1);
    const auto xsegm1 = eval.m_x.segment(1, size() - 1);

    if (eval.has_grad())
    {
        const auto weight = m_bias.segment(1, size() - 1).array() * 2 * (2 * xsegm1.array().square() - xsegm0.array());

        eval.m_gx.full(0);
        eval.m_gx(0) = 2 * (eval.m_x(0) - 1);
        eval.m_gx.segment(1, size() - 1).array() += weight * 4 * xsegm1.array();
        eval.m_gx.segment(0, size() - 1).array() -= weight;
    }

    if (eval.has_hess())
    {
        // FIXME:
        eval.m_Hx.full(0.0);
    }

    return nano::square(eval.m_x(0) - 1) +
           (m_bias.segment(1, size() - 1).array() * (2 * xsegm1.array().square() - xsegm0.array()).square()).sum();
}

rfunction_t function_dixon_price_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_dixon_price_t>(dims);
}
