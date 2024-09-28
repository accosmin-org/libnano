#include <function/benchmark/chung_reynolds.h>

using namespace nano;

function_chung_reynolds_t::function_chung_reynolds_t(const tensor_size_t dims)
    : function_t("chung-reynolds", dims)
{
    convex(convexity::yes);
    smooth(smoothness::yes);
}

rfunction_t function_chung_reynolds_t::clone() const
{
    return std::make_unique<function_chung_reynolds_t>(*this);
}

scalar_t function_chung_reynolds_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto u = x.dot(x);

    if (gx.size() == x.size())
    {
        gx = (4 * u) * x.vector();
    }

    return u * u;
}

rfunction_t function_chung_reynolds_t::make(const tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_chung_reynolds_t>(dims);
}
