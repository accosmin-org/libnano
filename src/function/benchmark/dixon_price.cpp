#include <nano/core/numeric.h>
#include <nano/function/benchmark/dixon_price.h>

using namespace nano;

function_dixon_price_t::function_dixon_price_t(tensor_size_t dims)
    : benchmark_function_t("Dixon-Price", dims)
    , m_bias(vector_t::LinSpaced(dims, scalar_t(1), scalar_t(dims)))
{
    convex(false);
    smooth(true);
}

scalar_t function_dixon_price_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    const auto xsegm0 = x.segment(0, size() - 1);
    const auto xsegm1 = x.segment(1, size() - 1);

    if (gx != nullptr)
    {
        const auto weight = m_bias.segment(1, size() - 1).array() * 2 * (2 * xsegm1.array().square() - xsegm0.array());

        (*gx).setZero();
        (*gx)(0) = 2 * (x(0) - 1);
        (*gx).segment(1, size() - 1).array() += weight * 4 * xsegm1.array();
        (*gx).segment(0, size() - 1).array() -= weight;
    }

    return nano::square(x(0) - 1) +
           (m_bias.segment(1, size() - 1).array() * (2 * xsegm1.array().square() - xsegm0.array()).square()).sum();
}

rfunction_t function_dixon_price_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_dixon_price_t>(dims);
}
