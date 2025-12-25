#include <function/ml/enet_cp.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>

using namespace nano;

namespace
{
tensor_size_t make_size(const tensor_size_t dims)
{
    return 2 * std::max(dims, tensor_size_t{2});
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
enet_program_t<tloss>::enet_program_t(const tensor_size_t dims, const uint64_t seed, const scalar_t sratio,
                                      const tensor_size_t modulo, const scalar_t alpha1, const scalar_t alpha2)
    : function_t(scat(tloss::basename, "+enet+cp"), make_size(dims))
    , m_dataset(make_samples(dims, sratio), make_outputs(dims), make_inputs(dims), seed, modulo, tloss::regression)
{
    this->register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    this->register_parameter(parameter_t::make_scalar("function::enet+cp::alpha1", 0.0, LE, alpha1, LE, 1e+8));
    this->register_parameter(parameter_t::make_scalar("function::enet+cp::alpha2", 0.0, LE, alpha2, LE, 1e+8));
    this->register_parameter(parameter_t::make_scalar("function::enet+cp::sratio", 0.1, LE, sratio, LE, 1e+3));
    this->register_parameter(parameter_t::make_integer("function::enet+cp::modulo", 1, LE, modulo, LE, 100));

    function_t::convex(tloss::convex ? convexity::yes : convexity::no);
    function_t::strong_convexity(alpha2);
    function_t::smooth(tloss::smooth ? smoothness::yes : smoothness::no);

    // min  f(x, z)
    // s.t. +x <= z
    //      -x <= z
    const auto n        = size() / 2;
    auto       A        = matrix_t{2 * n, 2 * n};
    A.block(0, 0, n, n) = matrix_t::identity(n, n);
    A.block(0, n, n, n) = -matrix_t::identity(n, n);
    A.block(n, 0, n, n) = -matrix_t::identity(n, n);
    A.block(n, n, n, n) = -matrix_t::identity(n, n);

    critical(A * variable() <= vector_t::zero(2 * n));
}

template <class tloss>
rfunction_t enet_program_t<tloss>::clone() const
{
    return std::make_unique<enet_program_t<tloss>>(*this);
}

template <class tloss>
string_t enet_program_t<tloss>::do_name() const
{
    const auto seed   = this->parameter("function::seed").template value<uint64_t>();
    const auto alpha1 = this->parameter("function::enet+cp::alpha1").template value<scalar_t>();
    const auto alpha2 = this->parameter("function::enet+cp::alpha2").template value<scalar_t>();
    const auto sratio = this->parameter("function::enet+cp::sratio").template value<scalar_t>();
    const auto modulo = this->parameter("function::enet+cp::modulo").template value<tensor_size_t>();

    return scat(this->type_id(), "[alpha1=", alpha1, ",alpha2=", alpha2, ",sratio=", sratio, ",modulo=", modulo,
                ",seed=", seed, "]");
}

template <class tloss>
scalar_t enet_program_t<tloss>::do_eval(function_t::eval_t eval) const
{
    const auto alpha1 = this->parameter("function::enet+cp::alpha1").template value<scalar_t>();
    const auto alpha2 = this->parameter("function::enet+cp::alpha2").template value<scalar_t>();

    const auto n = size() / 2;
    const auto x = eval.m_x.segment(0, n);
    const auto z = eval.m_x.segment(n, n);

    auto gx = eval.m_gx;
    auto hx = eval.m_hx;

    auto fx = m_dataset.do_eval<tloss>({
        .m_x  = eval.m_x.slice(0, n),
        .m_gx = eval.has_grad() ? gx.slice(0, n) : gx.tensor(),
        .m_hx = eval.has_hess() ? hx.reshape(hx.size()).slice(0, n * n).reshape(n, n).tensor() : hx.tensor(),
    });

    if (eval.has_grad())
    {
        gx.segment(0, n).array() += alpha2 * x.array();
        gx.segment(n, n).array() = alpha1;
    }

    if (eval.has_hess())
    {
        hx.block(n, n, n, n) = hx.reshape(4 * n * n).slice(0, n * n).reshape(n, n).matrix();
        hx.block(0, 0, n, n) = hx.block(n, n, n, n);
        hx.block(0, n, n, n) = matrix_t::zero(n, n);
        hx.block(n, 0, n, n) = matrix_t::zero(n, n);
        hx.block(n, n, n, n) = matrix_t::zero(n, n);
        hx.block(0, 0, n, n).diagonal().array() += alpha2;
    }

    fx += alpha1 * z.sum() + 0.5 * (std::sqrt(alpha2) * x).squaredNorm();
    return fx;
}

template <class tloss>
rfunction_t enet_program_t<tloss>::make(const tensor_size_t dims) const
{
    const auto seed   = this->parameter("function::seed").template value<uint64_t>();
    const auto alpha1 = this->parameter("function::enet+cp::alpha1").template value<scalar_t>();
    const auto alpha2 = this->parameter("function::enet+cp::alpha2").template value<scalar_t>();
    const auto sratio = this->parameter("function::enet+cp::sratio").template value<scalar_t>();
    const auto modulo = this->parameter("function::enet+cp::modulo").template value<tensor_size_t>();

    return std::make_unique<enet_program_t<tloss>>(dims / 2, seed, sratio, modulo, alpha1, alpha2);
}

template class nano::enet_program_t<nano::loss_mse_t>;
template class nano::enet_program_t<nano::loss_mae_t>;
template class nano::enet_program_t<nano::loss_hinge_t>;
template class nano::enet_program_t<nano::loss_cauchy_t>;
template class nano::enet_program_t<nano::loss_logistic_t>;
