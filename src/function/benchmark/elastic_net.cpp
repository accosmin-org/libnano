#include <function/benchmark/elastic_net.h>
#include <nano/core/strutil.h>

using namespace nano;

namespace
{
auto make_suffix(const scalar_t alpha1, const scalar_t alpha2)
{
    assert(alpha1 >= 0.0);
    assert(alpha2 >= 0.0);

    if (alpha1 == 0.0)
    {
        return scat("ridge[", alpha2, "]");
    }
    else if (alpha2 == 0.0)
    {
        return scat("lasso[", alpha1, "]");
    }
    else
    {
        return scat("elasticnet[", alpha1, ",", alpha2, "]");
    }
}

auto make_size(const tensor_size_t dims)
{
    return std::max(dims, tensor_size_t{2});
}

auto make_inputs(const tensor_size_t dims)
{
    return std::max(dims, tensor_size_t{2});
}

auto make_outputs(const tensor_size_t)
{
    return tensor_size_t{1};
}
} // namespace

template <class tloss>
function_enet_t<tloss>::function_enet_t(const tensor_size_t dims, const scalar_t alpha1, const scalar_t alpha2,
                                        const tensor_size_t summands)
    : function_t(scat(tloss::basename, "+", make_suffix(alpha1, alpha2)), ::make_size(dims))
    , tloss(summands, make_outputs(dims), make_inputs(dims))
    , m_alpha1(alpha1)
    , m_alpha2(alpha2)
{
    function_t::convex(tloss::convex ? convexity::yes : convexity::no);
    function_t::smooth((m_alpha1 == 0.0 && tloss::smooth) ? smoothness::yes : smoothness::no);
    strong_convexity(m_alpha2);
}

template <class tloss>
rfunction_t function_enet_t<tloss>::clone() const
{
    return std::make_unique<function_enet_t<tloss>>(*this);
}

template <class tloss>
scalar_t function_enet_t<tloss>::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto targets = this->targets();
    const auto outputs = this->outputs(x);

    auto fx = tloss::vgrad(outputs, targets, gx);

    if (gx.size() == x.size())
    {
        gx.array() += m_alpha1 * x.array().sign() + m_alpha2 * x.array();
    }

    fx += m_alpha1 * x.template lpNorm<1>() + 0.5 * (std::sqrt(m_alpha2) * x).squaredNorm();
    return fx;
}

template <class tloss>
rfunction_t function_enet_t<tloss>::make(const tensor_size_t dims, const tensor_size_t summands) const
{
    return std::make_unique<function_enet_t<tloss>>(dims, m_alpha1, m_alpha2, summands);
}

template class nano::function_enet_t<nano::loss_mse_t>;
template class nano::function_enet_t<nano::loss_mae_t>;
template class nano::function_enet_t<nano::loss_hinge_t>;
template class nano::function_enet_t<nano::loss_cauchy_t>;
template class nano::function_enet_t<nano::loss_logistic_t>;
