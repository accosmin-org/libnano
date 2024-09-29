#include <function/benchmark/maxhilb.h>

using namespace nano;

function_maxhilb_t::function_maxhilb_t(const tensor_size_t dims)
    : function_t("maxhilb", dims)
    , m_weights(dims, dims)
{
    convex(convexity::yes);
    smooth(smoothness::no);
    strong_convexity(0.0);

    for (tensor_size_t i = 0; i < dims; ++i)
    {
        for (tensor_size_t j = 0; j < dims; ++j)
        {
            m_weights(i, j) = 1.0 / static_cast<scalar_t>(i + j + 1);
        }
    }
}

rfunction_t function_maxhilb_t::clone() const
{
    return std::make_unique<function_maxhilb_t>(*this);
}

scalar_t function_maxhilb_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    auto       idx = tensor_size_t{0};
    const auto fx  = (m_weights * x).array().abs().maxCoeff(&idx);

    if (gx.size() == x.size())
    {
        const auto wei = m_weights.row(idx).transpose();
        gx             = wei * (std::signbit(x.dot(wei)) ? -1.0 : +1.0);
    }

    return fx;
}

rfunction_t function_maxhilb_t::make(const tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_maxhilb_t>(dims);
}
