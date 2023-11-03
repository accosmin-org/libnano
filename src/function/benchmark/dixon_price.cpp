#include <nano/core/numeric.h>
#include <nano/function/benchmark/dixon_price.h>

using namespace nano;

function_dixon_price_t::function_dixon_price_t(tensor_size_t dims)
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
    const auto xv     = x.vector();
    const auto bv     = m_bias.vector();
    const auto xsegm0 = xv.segment(0, size() - 1);
    const auto xsegm1 = xv.segment(1, size() - 1);

    if (gx.size() == x.size())
    {
        const auto weight = bv.segment(1, size() - 1).array() * 2 * (2 * xsegm1.array().square() - xsegm0.array());

        gx.full(0);
        gx(0) = 2 * (x(0) - 1);
        gx.vector().segment(1, size() - 1).array() += weight * 4 * xsegm1.array();
        gx.vector().segment(0, size() - 1).array() -= weight;
    }

    return nano::square(x(0) - 1) +
           (bv.segment(1, size() - 1).array() * (2 * xsegm1.array().square() - xsegm0.array()).square()).sum();
}

rfunction_t function_dixon_price_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_dixon_price_t>(dims);
}
