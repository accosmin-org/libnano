#include <function/ml/lasso.h>

using namespace nano;

template <class tloss>
lasso_function_t<tloss>::lasso_function_t(const tensor_size_t dims, const uint64_t seed, const scalar_t sratio,
                                          const tensor_size_t modulo, const lasso_type type, const scalar_t alpha1)
    : linear_model_t<tloss>("lasso", dims, seed, sratio, modulo, type, alpha1, 0.0)
{
    this->register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    this->register_parameter(parameter_t::make_scalar("function::lasso::alpha1", 0.0, LE, alpha1, LE, 1e+8));
    this->register_parameter(parameter_t::make_scalar("function::lasso::sratio", 0.1, LE, sratio, LE, 1e+3));
    this->register_parameter(parameter_t::make_integer("function::lasso::modulo", 1, LE, modulo, LE, 100));
    this->register_parameter(parameter_t::make_enum("function::lasso::type", type));
}

template <class tloss>
rfunction_t lasso_function_t<tloss>::clone() const
{
    return std::make_unique<lasso_function_t<tloss>>(*this);
}

template <class tloss>
string_t lasso_function_t<tloss>::do_name() const
{
    const auto seed   = this->parameter("function::seed").template value<uint64_t>();
    const auto alpha1 = this->parameter("function::lasso::alpha1").template value<scalar_t>();
    const auto sratio = this->parameter("function::lasso::sratio").template value<scalar_t>();
    const auto modulo = this->parameter("function::lasso::modulo").template value<tensor_size_t>();

    return scat(this->type_id(), "[alpha1=", alpha1, ",sratio=", sratio, ",modulo=", modulo, ",seed=", seed, "]");
}

template <class tloss>
scalar_t lasso_function_t<tloss>::do_eval(function_t::eval_t eval) const
{
    const auto type   = this->parameter("function::lasso::type").template value<lasso_type>();
    const auto alpha1 = this->parameter("function::lasso::alpha1").template value<scalar_t>();
    const auto alpha2 = 0.0;

    return this->do_enet_eval(eval, type, alpha1, alpha2);
}

template <class tloss>
rfunction_t lasso_function_t<tloss>::make(const tensor_size_t dims) const
{
    const auto seed   = this->parameter("function::seed").template value<uint64_t>();
    const auto alpha1 = this->parameter("function::lasso::alpha1").template value<scalar_t>();
    const auto sratio = this->parameter("function::lasso::sratio").template value<scalar_t>();
    const auto modulo = this->parameter("function::lasso::modulo").template value<tensor_size_t>();
    const auto type   = this->parameter("function::lasso::type").template value<lasso_type>();

    return std::make_unique<lasso_function_t<tloss>>((type == lasso_type::constrained) ? (dims / 2) : dims, seed,
                                                     sratio, modulo, type, alpha1);
}

template class nano::lasso_function_t<nano::loss_mse_t>;
template class nano::lasso_function_t<nano::loss_mae_t>;
template class nano::lasso_function_t<nano::loss_hinge_t>;
template class nano::lasso_function_t<nano::loss_cauchy_t>;
template class nano::lasso_function_t<nano::loss_logistic_t>;
