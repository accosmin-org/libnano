#include <nano/core/numeric.h>
#include <nano/function/zakharov.h>

using namespace nano;

function_zakharov_t::function_zakharov_t(tensor_size_t dims) :
    benchmark_function_t("Zakharov", dims),
    m_bias(vector_t::LinSpaced(dims, scalar_t(0.5), scalar_t(dims) / scalar_t(2)))
{
    convex(true);
    smooth(true);
}

scalar_t function_zakharov_t::vgrad(const vector_t& x, vector_t* gx) const
{
    const scalar_t u = x.dot(x);
    const scalar_t v = x.dot(m_bias);

    if (gx != nullptr)
    {
        *gx = 2 * x + (2 * v + 4 * nano::cube(v)) * m_bias;
    }

    return u + nano::square(v) + nano::quartic(v);
}

rfunction_t function_zakharov_t::make(tensor_size_t dims) const
{
    return std::make_unique<function_zakharov_t>(dims);
}
