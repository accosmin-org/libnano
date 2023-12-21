#include <nano/program/solver.h>
#include <nano/solver/proximal.h>

using namespace nano;

namespace
{
struct sequence_t
{
    scalar_t update() { return m_lambda = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * m_lambda * m_lambda)); }

    scalar_t m_lambda{1.0};
};
} // namespace

struct proximal::sequence1_t final : public sequence_t
{
    auto alpha_beta()
    {
        const auto curr  = m_lambda;
        const auto next  = update();
        const auto alpha = (curr - 1.0) / next;
        const auto beta  = 0.0;
        return std::make_tuple(alpha, beta);
    }
};

struct proximal::sequence2_t final : public sequence_t
{
    auto alpha_beta()
    {
        const auto curr  = m_lambda;
        const auto next  = update();
        const auto alpha = (curr - 1.0) / next;
        const auto beta  = curr / next;
        return std::make_tuple(alpha, beta);
    }
};

struct proximal::fpba1_type_id_t
{
    static auto str() { return "fpba1"; }
};

struct proximal::fpba2_type_id_t
{
    static auto str() { return "fpba2"; }
};

template <typename tsequence, typename ttype_id>
base_solver_fpba_t<tsequence, ttype_id>::base_solver_fpba_t()
    : solver_t(ttype_id::str())
{
    type(solver_type::non_monotonic);

    const auto basename = scat("solver::", ttype_id::str(), "::");

    register_parameter(parameter_t::make_scalar(basename + "miu", 0, LT, 1.0, LT, 1e+6));
}

template <typename tsequence, typename ttype_id>
rsolver_t base_solver_fpba_t<tsequence, ttype_id>::clone() const
{
    return std::make_unique<base_solver_fpba_t<tsequence, ttype_id>>(*this);
}

template <typename tsequence, typename ttype_id>
solver_state_t base_solver_fpba_t<tsequence, ttype_id>::do_minimize(const function_t& function,
                                                                    const vector_t&   x0) const
{
    const auto basename  = scat("solver::", ttype_id::str(), "::");
    const auto max_evals = parameter("solver::max_evals").template value<tensor_size_t>();
    const auto epsilon   = parameter("solver::epsilon").template value<scalar_t>();
    const auto miu       = parameter(basename + "miu").template value<scalar_t>();

    (void)epsilon;
    (void)miu;

    auto state = solver_state_t{function, x0};

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        break;
    }

    return state;
}

template class nano::base_solver_fpba_t<proximal::sequence1_t, proximal::fpba1_type_id_t>;
template class nano::base_solver_fpba_t<proximal::sequence2_t, proximal::fpba2_type_id_t>;
