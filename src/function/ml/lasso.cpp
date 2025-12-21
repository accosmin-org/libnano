#include <function/ml/lasso.h>
#include <function/ml/util.h>
#include <nano/core/strutil.h>

using namespace nano;

template <class tloss>
lasso_function_t<tloss>::lasso_function_t(const tensor_size_t dims, const uint64_t seed, const scalar_t alpha1,
                                          const scalar_t sratio, const tensor_size_t modulo)
    : function_t(scat(tloss::basename, "+lasso"), ::make_size(dims))
    , m_model(make_samples(dims, sratio), make_outputs(dims), make_inputs(dims), seed, modulo, tloss::regression)
{
    register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    register_parameter(parameter_t::make_scalar("function::lasso::alpha1", 0.0, LE, alpha1, LE, 1e+8));
    register_parameter(parameter_t::make_scalar("function::lasso::sratio", 0.1, LE, sratio, LE, 1e+3));
    register_parameter(parameter_t::make_integer("function::lasso::modulo", 1, LE, modulo, LE, 100));

    function_t::convex(tloss::convex ? convexity::yes : convexity::no);
    function_t::smooth(smoothness::no);
    function_t::strong_convexity(0.0);
}

template <class tloss>
rfunction_t lasso_function_t<tloss>::clone() const
{
    return std::make_unique<lasso_function_t<tloss>>(*this);
}

template <class tloss>
string_t lasso_function_t<tloss>::do_name() const
{
    const auto seed   = parameter("function::seed").template value<uint64_t>();
    const auto alpha1 = parameter("function::lasso::alpha1").template value<scalar_t>();
    const auto sratio = parameter("function::lasso::sratio").template value<scalar_t>();
    const auto modulo = parameter("function::lasso::modulo").template value<tensor_size_t>();

    return scat(type_id(), "[alpha1=", alpha1, ",sratio=", sratio, ",modulo=", modulo, ",seed=", seed, "]");
}

template <class tloss>
scalar_t lasso_function_t<tloss>::do_eval(eval_t eval) const
{
    const auto alpha1 = parameter("function::lasso::alpha1").template value<scalar_t>();

    auto fx = m_model.eval<tloss>(eval);

    if (eval.has_grad())
    {
        eval.m_gx.array() += alpha1 * eval.m_x.array().sign();
    }

    fx += alpha1 * eval.m_x.template lpNorm<1>();
    return fx;
}

template <class tloss>
rfunction_t lasso_function_t<tloss>::make(const tensor_size_t dims) const
{
    const auto seed   = parameter("function::seed").template value<uint64_t>();
    const auto alpha1 = parameter("function::lasso::alpha1").template value<scalar_t>();
    const auto sratio = parameter("function::lasso::sratio").template value<scalar_t>();
    const auto modulo = parameter("function::lasso::modulo").template value<tensor_size_t>();

    return std::make_unique<lasso_function_t<tloss>>(dims, seed, alpha1, sratio, modulo);
}

template class nano::lasso_function_t<nano::loss_mse_t>;
template class nano::lasso_function_t<nano::loss_mae_t>;
template class nano::lasso_function_t<nano::loss_hinge_t>;
template class nano::lasso_function_t<nano::loss_cauchy_t>;
template class nano::lasso_function_t<nano::loss_logistic_t>;
