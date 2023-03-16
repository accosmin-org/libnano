#include <nano/solver/nonsmooth_state.h>

using namespace nano;

nonsmooth_solver_state_t::nonsmooth_solver_state_t(solver_state_t& state, const tensor_size_t patience)
    : m_state(state)
    , m_df_history(patience)
    , m_dx_history(patience)
{
    m_df_history.array() = 0.0;
    m_dx_history.array() = 0.0;
}

bool nonsmooth_solver_state_t::update_if_better(const vector_t& x, const scalar_t fx)
{
    return update_if_better(x, m_state.g, fx);
}

bool nonsmooth_solver_state_t::update_if_better(const vector_t& x, const vector_t& gx, const scalar_t fx)
{
    const auto df      = m_state.f - fx;
    const auto dx      = (m_state.x - x).lpNorm<Eigen::Infinity>();
    const auto updated = m_state.update_if_better(x, gx, fx);

    m_df_history(m_iteration % m_df_history.size()) = updated ? df : 0.0;
    m_dx_history(m_iteration % m_dx_history.size()) = updated ? dx : 0.0;

    ++m_iteration;
    return updated;
}

bool nonsmooth_solver_state_t::converged(const scalar_t epsilon) const
{
    return m_iteration >= m_df_history.size() && m_df_history.sum() < epsilon && m_dx_history.sum() < epsilon;
}
