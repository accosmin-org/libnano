#include <function/benchmark/sargan.h>
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

scalar_t function_sargan_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto x2sum = x.dot(x);

    if (gx.size() == x.size())
    {
        gx = (1.2 + 1.6 * x2sum) * x;
    }

    return 0.6 * x2sum + 0.4 * nano::square(x2sum);
}

rfunction_t function_sargan_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_sargan_t>(dims);
}
