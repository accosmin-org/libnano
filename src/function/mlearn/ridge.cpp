#include <function/mlearn/ridge.h>
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
function_ridge_t<tloss>::function_ridge_t(const tensor_size_t dims, const uint64_t seed, const scalar_t alpha2,
                                          const scalar_t sratio, const tensor_size_t modulo)
    : function_t(scat(tloss::basename, "+ridge"), ::make_size(dims))
    , m_model(make_samples(dims, sratio), make_outputs(dims), make_inputs(dims), seed, modulo, tloss::regression)
{
    register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    register_parameter(parameter_t::make_scalar("function::ridge::alpha2", 0.0, LE, alpha2, LE, 1e+8));
    register_parameter(parameter_t::make_scalar("function::ridge::sratio", 0.1, LE, sratio, LE, 1e+3));
    register_parameter(parameter_t::make_integer("function::ridge::modulo", 1, LE, modulo, LE, 100));

    function_t::convex(tloss::convex ? convexity::yes : convexity::no);
    function_t::smooth(tloss::smooth ? smoothness::yes : smoothness::no);
    function_t::strong_convexity(alpha2);
}

template <class tloss>
rfunction_t function_ridge_t<tloss>::clone() const
{
    return std::make_unique<function_ridge_t<tloss>>(*this);
}

template <class tloss>
string_t function_ridge_t<tloss>::do_name() const
{
    const auto seed   = parameter("function::seed").template value<uint64_t>();
    const auto alpha2 = parameter("function::ridge::alpha2").template value<scalar_t>();
    const auto sratio = parameter("function::ridge::sratio").template value<scalar_t>();
    const auto modulo = parameter("function::ridge::modulo").template value<tensor_size_t>();

    return scat(type_id(), "[alpha2=", alpha2, ",sratio=", sratio, ",modulo=", modulo, ",seed=", seed, "]");
}

template <class tloss>
scalar_t function_ridge_t<tloss>::do_eval(eval_t eval) const
{
    const auto alpha2 = parameter("function::ridge::alpha2").template value<scalar_t>();

    auto fx = tloss::eval(m_model.outputs(eval.m_x), m_model.targets(), m_model.gradients(), m_model.hessians());

    if (m_model.eval_grad(eval.m_gx))
    {
        eval.m_gx.array() += alpha2 * eval.m_x.array();
    }

    if (smooth() && m_model.eval_hess(eval.m_Hx))
    {
        eval.m_Hx.diagonal() += alpha2 * matrix_t::identity(size(), size());
    }

    fx += 0.5 * (std::sqrt(alpha2) * eval.m_x).squaredNorm();
    return fx;
}

template <class tloss>
rfunction_t function_ridge_t<tloss>::make(const tensor_size_t dims) const
{
    const auto seed   = parameter("function::seed").template value<uint64_t>();
    const auto alpha2 = parameter("function::ridge::alpha2").template value<scalar_t>();
    const auto sratio = parameter("function::ridge::sratio").template value<scalar_t>();
    const auto modulo = parameter("function::ridge::modulo").template value<tensor_size_t>();

    return std::make_unique<function_ridge_t<tloss>>(dims, seed, alpha2, sratio, modulo);
}

template class nano::function_ridge_t<nano::loss_mse_t>;
template class nano::function_ridge_t<nano::loss_mae_t>;
template class nano::function_ridge_t<nano::loss_hinge_t>;
template class nano::function_ridge_t<nano::loss_cauchy_t>;
template class nano::function_ridge_t<nano::loss_logistic_t>;
