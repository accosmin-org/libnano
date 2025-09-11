#include <function/nonlinear/zakharov.h>
#include <nano/core/numeric.h>

using namespace nano;

function_zakharov_t::function_zakharov_t(const tensor_size_t dims)
    : function_t("zakharov", dims)
    , m_bias(dims)
{
    m_bias.lin_spaced(scalar_t(0.5), scalar_t(dims) / scalar_t(2));

    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_zakharov_t::clone() const
{
    return std::make_unique<function_zakharov_t>(*this);
}

scalar_t function_zakharov_t::do_eval(eval_t eval) const
{
    const auto x = eval.m_x;
    const auto u = x.dot(x);
    const auto v = x.dot(m_bias);

    if (eval.has_grad())
    {
        eval.m_gx = 2 * x + (2 * v + 4 * nano::cube(v)) * m_bias;
    }

    if (eval.has_hess())
    {
        eval.m_Hx = (2 + 12 * v * v) * (m_bias.vector() * m_bias.transpose());
        eval.m_Hx.diagonal().array() += 2;
    }

    return u + nano::square(v) + nano::quartic(v);
}

rfunction_t function_zakharov_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_zakharov_t>(dims);
}
