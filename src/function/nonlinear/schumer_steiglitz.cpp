#include <function/nonlinear/schumer_steiglitz.h>

using namespace nano;

function_schumer_steiglitz_t::function_schumer_steiglitz_t(const tensor_size_t dims)
    : function_t("schumer-steiglitz", dims)
{
    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_schumer_steiglitz_t::clone() const
{
    return std::make_unique<function_schumer_steiglitz_t>(*this);
}

scalar_t function_schumer_steiglitz_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    if (gx.size() == x.size())
    {
        gx = 4 * x.array().cube();
    }

    return x.array().square().square().sum();
}

rfunction_t function_schumer_steiglitz_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_schumer_steiglitz_t>(dims);
}
