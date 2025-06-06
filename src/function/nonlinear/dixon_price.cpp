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

scalar_t function_dixon_price_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto xsegm0 = x.segment(0, size() - 1);
    const auto xsegm1 = x.segment(1, size() - 1);

    if (gx.size() == x.size())
    {
        const auto weight = m_bias.segment(1, size() - 1).array() * 2 * (2 * xsegm1.array().square() - xsegm0.array());

        gx.full(0);
        gx(0) = 2 * (x(0) - 1);
        gx.segment(1, size() - 1).array() += weight * 4 * xsegm1.array();
        gx.segment(0, size() - 1).array() -= weight;
    }

    return nano::square(x(0) - 1) +
           (m_bias.segment(1, size() - 1).array() * (2 * xsegm1.array().square() - xsegm0.array()).square()).sum();
}

rfunction_t function_dixon_price_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_dixon_price_t>(dims);
}
