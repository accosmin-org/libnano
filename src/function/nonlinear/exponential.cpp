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

scalar_t function_exponential_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto alpha = 1.0 / static_cast<scalar_t>(size());

    const auto fx = std::exp(1 + x.dot(x) * alpha);

    if (gx.size() == x.size())
    {
        gx = (2 * fx * alpha) * x.vector();
    }

    return fx;
}

rfunction_t function_exponential_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_exponential_t>(dims);
}
