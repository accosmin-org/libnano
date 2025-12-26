#include <function/ml/enet.h>

using namespace nano;

namespace
{
tensor_size_t make_size(const tensor_size_t dims)
{
    return std::max(dims, tensor_size_t{2});
}

tensor_size_t make_inputs(const tensor_size_t dims)
{
    return std::max(dims, tensor_size_t{2});
}

tensor_size_t make_outputs([[maybe_unused]] const tensor_size_t dims)
{
    return tensor_size_t{1};
}

tensor_size_t make_samples(const tensor_size_t dims, const scalar_t sratio)
{
    return static_cast<tensor_size_t>(std::max(sratio * static_cast<scalar_t>(dims), 10.0));
}
} // namespace

template <class tloss>
enet_function_t<tloss>::enet_function_t(const tensor_size_t dims, const uint64_t seed, const scalar_t sratio,
                                        const tensor_size_t modulo, const scalar_t alpha1, const scalar_t alpha2)
    : function_t(scat(tloss::basename, "+enet"), make_size(dims))
    , m_dataset(make_samples(dims, sratio), make_outputs(dims), make_inputs(dims), seed, modulo, tloss::regression)
{
    this->register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    this->register_parameter(parameter_t::make_scalar("function::enet::alpha1", 0.0, LE, alpha1, LE, 1e+8));
    this->register_parameter(parameter_t::make_scalar("function::enet::alpha2", 0.0, LE, alpha2, LE, 1e+8));
    this->register_parameter(parameter_t::make_scalar("function::enet::sratio", 0.1, LE, sratio, LE, 1e+3));
    this->register_parameter(parameter_t::make_integer("function::enet::modulo", 1, LE, modulo, LE, 100));

    function_t::convex(tloss::convex ? convexity::yes : convexity::no);
    function_t::strong_convexity(alpha2);
    function_t::smooth((alpha1 == 0.0 && tloss::smooth) ? smoothness::yes : smoothness::no);
}

template <class tloss>
rfunction_t enet_function_t<tloss>::clone() const
{
    return std::make_unique<enet_function_t<tloss>>(*this);
}

template <class tloss>
string_t enet_function_t<tloss>::do_name() const
{
    const auto seed   = this->parameter("function::seed").template value<uint64_t>();
    const auto alpha1 = this->parameter("function::enet::alpha1").template value<scalar_t>();
    const auto alpha2 = this->parameter("function::enet::alpha2").template value<scalar_t>();
    const auto sratio = this->parameter("function::enet::sratio").template value<scalar_t>();
    const auto modulo = this->parameter("function::enet::modulo").template value<tensor_size_t>();

    return scat(this->type_id(), "[alpha1=", alpha1, ",alpha2=", alpha2, ",sratio=", sratio, ",modulo=", modulo,
                ",seed=", seed, "]");
}

template <class tloss>
scalar_t enet_function_t<tloss>::do_eval(function_t::eval_t eval) const
{
    const auto alpha1 = this->parameter("function::enet::alpha1").template value<scalar_t>();
    const auto alpha2 = this->parameter("function::enet::alpha2").template value<scalar_t>();

    const auto n = size();
    const auto x = eval.m_x.segment(0, n);

    auto fx = m_dataset.do_eval<tloss>(eval);

    if (eval.has_grad())
    {
        eval.m_gx.array() += alpha1 * x.array().sign() + alpha2 * x.array();
    }

    if (eval.has_hess())
    {
        eval.m_hx.diagonal().array() += alpha2;
    }

    return fx + alpha1 * x.template lpNorm<1>() + 0.5 * (std::sqrt(alpha2) * x).squaredNorm();
}

template <class tloss>
rfunction_t enet_function_t<tloss>::make(const tensor_size_t dims) const
{
    const auto seed   = this->parameter("function::seed").template value<uint64_t>();
    const auto alpha1 = this->parameter("function::enet::alpha1").template value<scalar_t>();
    const auto alpha2 = this->parameter("function::enet::alpha2").template value<scalar_t>();
    const auto sratio = this->parameter("function::enet::sratio").template value<scalar_t>();
    const auto modulo = this->parameter("function::enet::modulo").template value<tensor_size_t>();

    return std::make_unique<enet_function_t<tloss>>(dims, seed, sratio, modulo, alpha1, alpha2);
}

template class nano::enet_function_t<nano::loss_mse_t>;
template class nano::enet_function_t<nano::loss_mae_t>;
template class nano::enet_function_t<nano::loss_hinge_t>;
template class nano::enet_function_t<nano::loss_cauchy_t>;
template class nano::enet_function_t<nano::loss_logistic_t>;
