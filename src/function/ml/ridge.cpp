#include <function/ml/ridge.h>

using namespace nano;

template <class tloss>
ridge_function_t<tloss>::ridge_function_t(const tensor_size_t dims, const uint64_t seed, const scalar_t sratio,
                                          const tensor_size_t modulo, const scalar_t alpha2)
    : linear_model_t<tloss>("ridge", dims, seed, sratio, modulo, lasso_type::unconstrained, 0.0, alpha2)
{
    this->register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    this->register_parameter(parameter_t::make_scalar("function::ridge::alpha2", 0.0, LE, alpha2, LE, 1e+8));
    this->register_parameter(parameter_t::make_scalar("function::ridge::sratio", 0.1, LE, sratio, LE, 1e+3));
    this->register_parameter(parameter_t::make_integer("function::ridge::modulo", 1, LE, modulo, LE, 100));
}

template <class tloss>
rfunction_t ridge_function_t<tloss>::clone() const
{
    return std::make_unique<ridge_function_t<tloss>>(*this);
}

template <class tloss>
string_t ridge_function_t<tloss>::do_name() const
{
    const auto seed   = this->parameter("function::seed").template value<uint64_t>();
    const auto alpha2 = this->parameter("function::ridge::alpha2").template value<scalar_t>();
    const auto sratio = this->parameter("function::ridge::sratio").template value<scalar_t>();
    const auto modulo = this->parameter("function::ridge::modulo").template value<tensor_size_t>();

    return scat(this->type_id(), "[alpha2=", alpha2, ",sratio=", sratio, ",modulo=", modulo, ",seed=", seed, "]");
}

template <class tloss>
scalar_t ridge_function_t<tloss>::do_eval(function_t::eval_t eval) const
{
    const auto type   = lasso_type::unconstrained;
    const auto alpha1 = 0.0;
    const auto alpha2 = this->parameter("function::ridge::alpha2").template value<scalar_t>();

    return this->do_enet_eval(eval, type, alpha1, alpha2);
}

template <class tloss>
rfunction_t ridge_function_t<tloss>::make(const tensor_size_t dims) const
{
    const auto seed   = this->parameter("function::seed").template value<uint64_t>();
    const auto alpha2 = this->parameter("function::ridge::alpha2").template value<scalar_t>();
    const auto sratio = this->parameter("function::ridge::sratio").template value<scalar_t>();
    const auto modulo = this->parameter("function::ridge::modulo").template value<tensor_size_t>();

    return std::make_unique<ridge_function_t<tloss>>(dims, seed, sratio, modulo, alpha2);
}

template class nano::ridge_function_t<nano::loss_mse_t>;
template class nano::ridge_function_t<nano::loss_mae_t>;
template class nano::ridge_function_t<nano::loss_hinge_t>;
template class nano::ridge_function_t<nano::loss_cauchy_t>;
template class nano::ridge_function_t<nano::loss_logistic_t>;
