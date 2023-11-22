#include <nano/function/benchmark/maxhilb.h>

using namespace nano;

function_maxhilb_t::function_maxhilb_t(tensor_size_t dims)
    : function_t("maxhilb", dims)
{
    convex(convexity::yes);
    smooth(smoothness::no);
    strong_convexity(0.0);
}

rfunction_t function_maxhilb_t::clone() const
{
    return std::make_unique<function_maxhilb_t>(*this);
}

scalar_t function_maxhilb_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    auto idx = tensor_size_t{0};
    auto fx  = std::numeric_limits<scalar_t>::lowest();
    auto sgn = +1.0;

    for (tensor_size_t i = 0, dims = x.size(); i < dims; ++i)
    {
        auto fi = 0.0;
        for (tensor_size_t j = 0; j < dims; ++j)
        {
            fi += x(j) / static_cast<scalar_t>(i + j + 1);
        }

        if (std::fabs(fi) > fx)
        {
            fx  = std::fabs(fi);
            idx = i;
            sgn = std::signbit(fi) ? -1.0 : +1.0;
        }
    }

    if (gx.size() == x.size())
    {
        for (tensor_size_t j = 0, dims = x.size(); j < dims; ++j)
        {
            gx(j) = sgn / static_cast<scalar_t>(idx + j + 1);
        }
    }

    return fx;
}

rfunction_t function_maxhilb_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_maxhilb_t>(dims);
}
