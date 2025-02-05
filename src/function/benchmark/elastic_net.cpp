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
function_enet_t<tloss>::function_enet_t(const tensor_size_t dims)
    : function_t(scat(tloss::basename, "+", make_suffix(0.0, 0.0)), ::make_size(dims))
    , m_model(100, make_outputs(dims), make_inputs(dims), 1, tloss::regression)
{
    register_parameter(parameter_t::make_scalar("enet::alpha1", 0.0, LE, 0.0, LE, 1e+8));
    register_parameter(parameter_t::make_scalar("enet::alpha2", 0.0, LE, 0.0, LE, 1e+8));
    register_parameter(parameter_t::make_scalar("enet::sample_ratio", 0.1, LE, 10.0, LE, 1e+3));
    register_parameter(parameter_t::make_integer("enet::modulo_correlated", 1, LE, 1, LE, 100));

    function_t::convex(tloss::convex ? convexity::yes : convexity::no);
    function_t::smooth(tloss::smooth ? smoothness::yes : smoothness::no);
    strong_convexity(0.0);
}

template <class tloss>
rfunction_t function_enet_t<tloss>::clone() const
{
    return std::make_unique<function_enet_t<tloss>>(*this);
}

template <class tloss>
scalar_t function_enet_t<tloss>::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto alpha1 = parameter("enet::alpha1").value<scalar_t>();
    const auto alpha2 = parameter("enet::alpha2").value<scalar_t>();

    auto fx = tloss::vgrad(m_model, m_model.outputs(x), m_model.targets(), gx);

    if (gx.size() == x.size())
    {
        gx.array() += alpha1 * x.array().sign() + alpha2 * x.array();
    }

    fx += alpha1 * x.template lpNorm<1>() + 0.5 * (std::sqrt(alpha2) * x).squaredNorm();
    return fx;
}

template <class tloss>
bool function_enet_t<tloss>::resize(const tensor_size_t dims)
{
    const auto alpha1 = parameter("enet::alpha1").value<scalar_t>();
    const auto alpha2 = parameter("enet::alpha2").value<scalar_t>();
    const auto sratio = parameter("enet::sample_ratio").value<scalar_t>();
    const auto modulo = parameter("enet::modulo_correlated").value<tensor_size_t>();

    const auto samples = static_cast<tensor_size_t>(std::max(sratio * static_cast<scalar_t>(dims), 10.0));

    function_t::rename(scat(tloss::basename, "+", make_suffix(alpha1, alpha2)), make_size(dims));
    function_t::smooth((alpha1 == 0.0 && tloss::smooth) ? smoothness::yes : smoothness::no);
    function_t::strong_convexity(alpha2);

    m_model = linear_model_t{samples, make_outputs(dims), make_inputs(dims), modulo, tloss::regression};

    return true;
}

template class nano::function_enet_t<nano::loss_mse_t>;
template class nano::function_enet_t<nano::loss_mae_t>;
template class nano::function_enet_t<nano::loss_hinge_t>;
template class nano::function_enet_t<nano::loss_cauchy_t>;
template class nano::function_enet_t<nano::loss_logistic_t>;
