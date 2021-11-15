#include <nano/core/strutil.h>
#include <nano/function/elastic_net.h>

using namespace nano;

static auto make_suffix(scalar_t alpha1, scalar_t alpha2)
{
    if (alpha1 == 0.0)
    {
        return "Ridge";
    }
    else
    {
        return alpha2 == 0.0 ? "Lasso": "ElasticNet";
    }
}

static auto make_size(tensor_size_t dims)
{
    return std::max(dims, tensor_size_t{2});
}

static auto make_inputs(tensor_size_t dims)
{
    return std::max(dims, tensor_size_t{2}) - 1;
}

static auto make_outputs(tensor_size_t)
{
    return tensor_size_t{1};
}

static auto make_samples(tensor_size_t dims)
{
    return 10 * std::max(dims, tensor_size_t{2});
}

template <typename tloss>
function_enet_t<tloss>::function_enet_t(tensor_size_t dims, scalar_t alpha1, scalar_t alpha2) :
    benchmark_function_t(scat(tloss::basename, "+", make_suffix(alpha1, alpha2)), ::make_size(dims)),
    tloss(dims),
    m_alpha1(alpha1),
    m_alpha2(alpha2)
{
    convex(true);
    smooth(m_alpha1 == 0.0 && tloss::smooth);
}

template <typename tloss>
scalar_t function_enet_t<tloss>::vgrad(const vector_t& x, vector_t* gx) const
{
    const auto fx = tloss::vgrad(x, gx);
    const auto w = tloss::make_w(x).matrix();

    if (gx != nullptr)
    {
        auto gw = tloss::make_w(*gx).matrix();

        gw.array() += m_alpha1 * w.array().sign();
        // cppcheck-suppress unreadVariable
        gw += m_alpha2 * w;
    }

    return fx + m_alpha1 * w.template lpNorm<1>() + 0.5 * m_alpha2 * w.squaredNorm();
}

template <typename tloss>
rfunction_t function_enet_t<tloss>::make(tensor_size_t dims) const
{
    return std::make_unique<function_enet_t<tloss>>(dims, m_alpha1, m_alpha2);
}

loss_mse_t::loss_mse_t(tensor_size_t dims) :
    synthetic_scalar_t(make_samples(dims), make_outputs(dims), make_inputs(dims))
{
}

scalar_t loss_mse_t::vgrad(const vector_t& x, vector_t* gx) const
{
    const auto outputs = this->outputs(x);
    const auto& targets = this->targets();

    const auto delta = outputs.matrix() - targets.matrix();
    const auto samples = static_cast<scalar_t>(this->samples());

    if (gx != nullptr)
    {
        synthetic_linear_t::vgrad(gx, delta);
    }

    return 0.5 * delta.squaredNorm() / samples;
}

loss_mae_t::loss_mae_t(tensor_size_t dims) :
    synthetic_scalar_t(make_samples(dims), make_outputs(dims), make_inputs(dims))
{
}

scalar_t loss_mae_t::vgrad(const vector_t& x, vector_t* gx) const
{
    const auto outputs = this->outputs(x);
    const auto& targets = this->targets();

    const auto delta = outputs.matrix() - targets.matrix();
    const auto samples = static_cast<scalar_t>(this->samples());

    if (gx != nullptr)
    {
        synthetic_linear_t::vgrad(gx, delta.array().sign().matrix());
    }

    return delta.array().abs().sum() / samples;
}

loss_hinge_t::loss_hinge_t(tensor_size_t dims) :
    synthetic_sclass_t(make_samples(dims), make_outputs(dims), make_inputs(dims))
{
}

scalar_t loss_hinge_t::vgrad(const vector_t& x, vector_t* gx) const
{
    const auto outputs = this->outputs(x);
    const auto& targets = this->targets();

    const auto edges = - outputs.array() * targets.array();
    const auto samples = static_cast<scalar_t>(this->samples());

    if (gx != nullptr)
    {
        synthetic_linear_t::vgrad(gx, (-targets.array() * ((1.0 + edges).sign() * 0.5 + 0.5)).matrix());
    }

    return (1.0 + edges).max(0.0).sum() / samples;
}

loss_logistic_t::loss_logistic_t(tensor_size_t dims) :
    synthetic_sclass_t(make_samples(dims), make_outputs(dims), make_inputs(dims))
{
}

scalar_t loss_logistic_t::vgrad(const vector_t& x, vector_t* gx) const
{
    const auto outputs = this->outputs(x);
    const auto& targets = this->targets();

    const auto edges = (-outputs.array() * targets.array()).exp();
    const auto samples = static_cast<scalar_t>(this->samples());

    if (gx != nullptr)
    {
        synthetic_linear_t::vgrad(gx, ((-targets.array() * edges) / (1.0 + edges)).matrix());
    }

    return (1.0 + edges).log().sum() / samples;
}

template class nano::function_enet_t<nano::loss_mse_t>;
template class nano::function_enet_t<nano::loss_mae_t>;
template class nano::function_enet_t<nano::loss_hinge_t>;
template class nano::function_enet_t<nano::loss_logistic_t>;
