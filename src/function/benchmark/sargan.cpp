#include <nano/core/numeric.h>
#include <nano/function/benchmark/sargan.h>

using namespace nano;

function_sargan_t::function_sargan_t(tensor_size_t dims)
    : function_t("sargan", dims)
{
    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_sargan_t::clone() const
{
    return std::make_unique<function_sargan_t>(*this);
}

scalar_t function_sargan_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    const auto x2sum = x.dot(x);

    if (gx != nullptr)
    {
        *gx = (scalar_t(1.2) + scalar_t(1.6) * x2sum) * x;
    }

    return scalar_t(0.6) * x2sum + scalar_t(0.4) * nano::square(x2sum);
}

rfunction_t function_sargan_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_sargan_t>(dims);
}
