#include <nano/program/state.h>

using namespace nano;
using namespace nano::program;

solver_state_t::solver_state_t() = default;

solver_state_t::solver_state_t(const tensor_size_t n, const tensor_size_t n_ineqs, const tensor_size_t n_eqs)
    : m_x(vector_t::constant(n, nan))
    , m_u(vector_t::constant(n_ineqs, nan))
    , m_v(vector_t::constant(n_eqs, nan))
    , m_rdual(vector_t::constant(n, nan))
    , m_rcent(vector_t::constant(n_ineqs, nan))
    , m_rprim(vector_t::constant(n_eqs, nan))
{
}

scalar_t solver_state_t::residual() const
{
    const auto edual = m_rdual.lpNorm<Eigen::Infinity>();
    const auto ecent = m_rcent.lpNorm<Eigen::Infinity>();
    const auto eprim = m_rprim.lpNorm<Eigen::Infinity>();
    return std::max({edual, ecent, eprim});
}

std::ostream& nano::program::operator<<(std::ostream& stream, const solver_state_t& state)
{
    return stream << "i=" << state.m_iters << ",fx=" << state.m_fx << ",eta=" << state.m_eta
                  << ",rdual=" << state.m_rdual.lpNorm<Eigen::Infinity>()
                  << ",rcent=" << state.m_rcent.lpNorm<Eigen::Infinity>()
                  << ",rprim=" << state.m_rprim.lpNorm<Eigen::Infinity>() << ",rcond=" << state.m_ldlt_rcond
                  << (state.m_ldlt_positive ? "(+)" : "(-)") << "[" << state.m_status << "]";
}
