#include <nano/function/benchmark/chainedlq.h>

using namespace nano;

function_chainedlq_t::function_chainedlq_t(tensor_size_t dims)
    : function_t("chainedlq", dims)
{
    convex(convexity::yes);
    smooth(smoothness::no);
    strong_convexity(0.0);
}

rfunction_t function_chainedlq_t::clone() const
{
    return std::make_unique<function_chainedlq_t>(*this);
}

scalar_t function_chainedlq_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    if (gx.size() == x.size())
    {
        gx.full(0.0);
        for (tensor_size_t i = 0, dims = size(); i + 1 < dims; ++i)
        {
            const auto xi  = x(i);
            const auto xi1 = x(i + 1);

            if (xi * xi + xi1 * xi1 >= 1.0)
            {
                gx(i) += -1.0 + 2 * xi;
                gx(i + 1) += -1.0 + 2 * xi1;
            }
            else
            {
                gx(i) += -1.0;
                gx(i + 1) += -1.0;
            }
        }
    }

    auto fx = 0.0;
    for (tensor_size_t i = 0, dims = size(); i + 1 < dims; ++i)
    {
        const auto xi  = x(i);
        const auto xi1 = x(i + 1);

        fx += std::max(-xi - xi1, -xi - xi1 + xi * xi + xi1 * xi1 - 1.0);
    }

    return fx;
}

rfunction_t function_chainedlq_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_chainedlq_t>(dims);
}
