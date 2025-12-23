#include <function/ml/elasticnet.h>

using namespace nano;

template <class tloss>
elasticnet_function_t<tloss>::elasticnet_function_t(const tensor_size_t dims, const uint64_t seed,
                                                    const scalar_t sratio, const tensor_size_t modulo,
                                                    const lasso_type type, const scalar_t alpha1, const scalar_t alpha2)
    : linear_model_t<tloss>("elasticnet", dims, seed, sratio, modulo, type, alpha1, alpha2)
{
    this->register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    this->register_parameter(parameter_t::make_scalar("function::elasticnet::alpha1", 0.0, LE, alpha1, LE, 1e+8));
    this->register_parameter(parameter_t::make_scalar("function::elasticnet::alpha2", 0.0, LE, alpha2, LE, 1e+8));
    this->register_parameter(parameter_t::make_scalar("function::elasticnet::sratio", 0.1, LE, sratio, LE, 1e+3));
    this->register_parameter(parameter_t::make_integer("function::elasticnet::modulo", 1, LE, modulo, LE, 100));
    this->register_parameter(parameter_t::make_enum("function::elasticnet::type", type));
}

template <class tloss>
rfunction_t elasticnet_function_t<tloss>::clone() const
{
    return std::make_unique<elasticnet_function_t<tloss>>(*this);
}

template <class tloss>
string_t elasticnet_function_t<tloss>::do_name() const
{
    const auto seed   = this->parameter("function::seed").template value<uint64_t>();
    const auto alpha1 = this->parameter("function::elasticnet::alpha1").template value<scalar_t>();
    const auto alpha2 = this->parameter("function::elasticnet::alpha2").template value<scalar_t>();
    const auto sratio = this->parameter("function::elasticnet::sratio").template value<scalar_t>();
    const auto modulo = this->parameter("function::elasticnet::modulo").template value<tensor_size_t>();

    return scat(this->type_id(), "[alpha1=", alpha1, ",alpha2=", alpha2, ",sratio=", sratio, ",modulo=", modulo,
                ",seed=", seed, "]");
}

template <class tloss>
scalar_t elasticnet_function_t<tloss>::do_eval(function_t::eval_t eval) const
{
    const auto type   = this->parameter("function::elasticnet::type").template value<lasso_type>();
    const auto alpha1 = this->parameter("function::elasticnet::alpha1").template value<scalar_t>();
    const auto alpha2 = this->parameter("function::elasticnet::alpha2").template value<scalar_t>();

    return this->do_enet_eval(eval, type, alpha1, alpha2);
}

template <class tloss>
rfunction_t elasticnet_function_t<tloss>::make(const tensor_size_t dims) const
{
    const auto seed   = this->parameter("function::seed").template value<uint64_t>();
    const auto alpha1 = this->parameter("function::elasticnet::alpha1").template value<scalar_t>();
    const auto alpha2 = this->parameter("function::elasticnet::alpha2").template value<scalar_t>();
    const auto sratio = this->parameter("function::elasticnet::sratio").template value<scalar_t>();
    const auto modulo = this->parameter("function::elasticnet::modulo").template value<tensor_size_t>();
    const auto type   = this->parameter("function::elasticnet::type").template value<lasso_type>();

    return std::make_unique<elasticnet_function_t<tloss>>((type == lasso_type::constrained) ? (dims / 2) : dims, seed,
                                                          sratio, modulo, type, alpha1, alpha2);
}

template class nano::elasticnet_function_t<nano::loss_mse_t>;
template class nano::elasticnet_function_t<nano::loss_mae_t>;
template class nano::elasticnet_function_t<nano::loss_hinge_t>;
template class nano::elasticnet_function_t<nano::loss_cauchy_t>;
template class nano::elasticnet_function_t<nano::loss_logistic_t>;
