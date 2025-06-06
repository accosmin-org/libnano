#include <function/nonlinear/chained_cb3I.h>
#include <nano/core/numeric.h>

using namespace nano;

function_chained_cb3I_t::function_chained_cb3I_t(const tensor_size_t dims)
    : function_t("chained_cb3I", dims)
{
    convex(convexity::yes);
    smooth(smoothness::no);
    strong_convexity(0.0);
}

rfunction_t function_chained_cb3I_t::clone() const
{
    return std::make_unique<function_chained_cb3I_t>(*this);
}

scalar_t function_chained_cb3I_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto values = [&](const tensor_size_t i = 0)
    {
        const auto xi  = x(i);
        const auto xi1 = x(i + 1);
        const auto v1  = quartic(xi) + square(xi1);
        const auto v2  = square(2.0 - xi) + square(2 - xi1);
        const auto v3  = 2.0 * std::exp(-xi + xi1);

        return std::make_tuple(v1, v2, v3);
    };

    if (gx.size() == x.size())
    {
        gx.full(0.0);
        for (tensor_size_t i = 0, dims = size(); i + 1 < dims; ++i)
        {
            const auto [v1, v2, v3] = values(i);

            if (v1 > std::max(v2, v3))
            {
                gx(i) += 4.0 * cube(x(i));
                gx(i + 1) += 2.0 * x(i + 1);
            }
            else if (v2 > std::max(v1, v3))
            {
                gx(i) -= 4.0 - 2.0 * x(i);
                gx(i + 1) -= 4.0 - 2.0 * x(i + 1);
            }
            else
            {
                const auto e = std::exp(x(i + 1) - x(i));
                gx(i) -= 2.0 * e;
                gx(i + 1) += 2.0 * e;
            }
        }
    }

    auto fx = 0.0;
    for (tensor_size_t i = 0, dims = size(); i + 1 < dims; ++i)
    {
        const auto [v1, v2, v3] = values(i);
        fx += std::max({v1, v2, v3});
    }

    return fx;
}

rfunction_t function_chained_cb3I_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_chained_cb3I_t>(dims);
}
