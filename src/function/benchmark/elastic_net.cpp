#include <nano/core/strutil.h>
#include <nano/function/benchmark/elastic_net.h>

using namespace nano;

static auto make_suffix(scalar_t alpha1, scalar_t alpha2)
{
    if (alpha1 == 0.0)
    {
        return "Ridge";
    }
    else
    {
        return alpha2 == 0.0 ? "Lasso" : "ElasticNet";
    }
}

static auto make_size(tensor_size_t dims)
{
    return std::max(dims, tensor_size_t{2});
}

static auto make_inputs(tensor_size_t dims)
{
    return std::max(dims, tensor_size_t{2});
}

static auto make_outputs(tensor_size_t)
{
    return tensor_size_t{1};
}

template <typename tloss>
function_enet_t<tloss>::function_enet_t(tensor_size_t dims, scalar_t alpha1, scalar_t alpha2, tensor_size_t summands)
    : benchmark_function_t(scat(tloss::basename, "+", make_suffix(alpha1, alpha2)), ::make_size(dims))
    , tloss(summands, make_outputs(dims), make_inputs(dims))
    , m_alpha1(alpha1)
    , m_alpha2(alpha2)
{
    convex(tloss::convex);
    smooth(m_alpha1 == 0.0 && tloss::smooth);
    strong_convexity(m_alpha2);
}

template <typename tloss>
scalar_t function_enet_t<tloss>::do_vgrad(const vector_t& x, vector_t* gx) const
{
    const auto inputs  = this->inputs();
    const auto targets = this->targets();
    const auto outputs = this->outputs(x);

    auto fx = tloss::vgrad(inputs, outputs, targets, gx);

    if (gx != nullptr)
    {
        gx->array() += m_alpha1 * x.array().sign() + m_alpha2 * x.array();
    }

    fx += m_alpha1 * x.template lpNorm<1>() + 0.5 * m_alpha2 * x.squaredNorm();
    return fx;
}

template <typename tloss>
rfunction_t function_enet_t<tloss>::make(tensor_size_t dims, tensor_size_t summands) const
{
    return std::make_unique<function_enet_t<tloss>>(dims, m_alpha1, m_alpha2, summands);
}

template class nano::function_enet_t<nano::loss_mse_t>;
template class nano::function_enet_t<nano::loss_mae_t>;
template class nano::function_enet_t<nano::loss_hinge_t>;
template class nano::function_enet_t<nano::loss_cauchy_t>;
template class nano::function_enet_t<nano::loss_logistic_t>;
