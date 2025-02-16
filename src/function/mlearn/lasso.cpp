#include <function/mlearn/lasso.h>
#include <nano/core/strutil.h>

using namespace nano;

namespace
{
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

auto make_samples(const tensor_size_t dims, const scalar_t sratio)
{
    return static_cast<tensor_size_t>(std::max(sratio * static_cast<scalar_t>(dims), 10.0));
}
} // namespace

template <class tloss>
function_lasso_t<tloss>::function_lasso_t(const tensor_size_t dims, const scalar_t alpha1, const scalar_t sratio,
                                          const tensor_size_t modulo)
    : function_t(scat(tloss::basename, "+lasso"), ::make_size(dims))
    , m_model(make_samples(dims, sratio), make_outputs(dims), make_inputs(dims), modulo, tloss::regression)
{
    register_parameter(parameter_t::make_scalar("lasso::alpha1", 0.0, LE, 0.0, LE, 1e+8));
    register_parameter(parameter_t::make_scalar("lasso::sratio", 0.1, LE, 10.0, LE, 1e+3));
    register_parameter(parameter_t::make_integer("lasso::modulo", 1, LE, 1, LE, 100));

    parameter("lasso::alpha1") = alpha1;
    parameter("lasso::sratio") = sratio;
    parameter("lasso::modulo") = modulo;

    function_t::convex(tloss::convex ? convexity::yes : convexity::no);
    function_t::smooth(smoothness::no);
    function_t::strong_convexity(0.0);
}

template <class tloss>
rfunction_t function_lasso_t<tloss>::clone() const
{
    return std::make_unique<function_lasso_t<tloss>>(*this);
}

template <class tloss>
string_t function_lasso_t<tloss>::do_name() const
{
    const auto alpha1 = parameter("lasso::alpha1").template value<scalar_t>();
    const auto sratio = parameter("lasso::sratio").template value<scalar_t>();
    const auto modulo = parameter("lasso::modulo").template value<tensor_size_t>();

    return scat(type_id(), "[alpha1=", alpha1, ",sratio=", sratio, ",modulo=", modulo, "]");
}

template <class tloss>
scalar_t function_lasso_t<tloss>::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto alpha1 = parameter("lasso::alpha1").template value<scalar_t>();

    auto fx = tloss::vgrad(m_model, m_model.outputs(x), m_model.targets(), gx);

    if (gx.size() == x.size())
    {
        gx.array() += alpha1 * x.array().sign();
    }

    fx += alpha1 * x.template lpNorm<1>();
    return fx;
}

template <class tloss>
rfunction_t function_lasso_t<tloss>::make(const tensor_size_t dims) const
{
    const auto alpha1 = parameter("lasso::alpha1").template value<scalar_t>();
    const auto sratio = parameter("lasso::sratio").template value<scalar_t>();
    const auto modulo = parameter("lasso::modulo").template value<tensor_size_t>();

    return std::make_unique<function_lasso_t<tloss>>(dims, alpha1, sratio, modulo);
}

template class nano::function_lasso_t<nano::loss_mse_t>;
template class nano::function_lasso_t<nano::loss_mae_t>;
template class nano::function_lasso_t<nano::loss_hinge_t>;
template class nano::function_lasso_t<nano::loss_cauchy_t>;
template class nano::function_lasso_t<nano::loss_logistic_t>;
