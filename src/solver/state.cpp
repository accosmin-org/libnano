#include <nano/function/constraint.h>
#include <nano/solver/state.h>

using namespace nano;

static auto fmax()
{
    return std::sqrt(std::numeric_limits<scalar_t>::max());
}

solver_state_t::solver_state_t() = default;

solver_state_t::solver_state_t(const function_t& function, vector_t x0)
    : m_function(&function)
    , m_x(std::move(x0))
    , m_gx(vector_t::Zero(m_x.size()))
    , m_fx(m_function->vgrad(m_x, &m_gx))
    , m_ceq(vector_t::Constant(::nano::count_equalities(m_function->constraints()), fmax()))
    , m_cineq(vector_t::Constant(::nano::count_inequalities(m_function->constraints()), fmax()))
{
    update_calls();
    update_constraints();
}

bool solver_state_t::update_if_better(const vector_t& x, const vector_t& gx, const scalar_t fx)
{
    update_calls();

    if (std::isfinite(fx))
    {
        const auto df = m_fx - fx;
        const auto dx = (m_x - x).lpNorm<Eigen::Infinity>();

        const auto better = df > 0.0;
        if (better)
        {
            m_x  = x;
            m_fx = fx;
            m_gx = gx;
            update_constraints();
        }

        m_history_df.push_back(df);
        m_history_dx.push_back(dx);

        return better;
    }
    else
    {
        m_history_df.push_back(std::numeric_limits<scalar_t>::lowest());
        m_history_dx.push_back(std::numeric_limits<scalar_t>::lowest());
        return false;
    }
}

bool solver_state_t::update_if_better(const vector_t& x, const scalar_t fx)
{
    return update_if_better(x, m_gx, fx);
}

bool solver_state_t::update_if_better_constrained(const solver_state_t& cstate, const scalar_t epsilon)
{
    auto updated = false;
    if (cstate.valid() && cstate.constraint_test() <= constraint_test() + epsilon)
    {
        updated = true;

        // NB: the original function value should be returned!
        update(cstate.m_x);
        m_status = cstate.m_status;
    }

    return updated;
}

void solver_state_t::update_calls()
{
    m_fcalls = m_function->fcalls();
    m_gcalls = m_function->gcalls();
}

void solver_state_t::update_constraints()
{
    tensor_size_t ieq = 0, ineq = 0;
    for (const auto& constraint : m_function->constraints())
    {
        if (::nano::is_equality(constraint))
        {
            m_ceq(ieq++) = ::vgrad(constraint, m_x);
        }
        else
        {
            m_cineq(ineq++) = ::vgrad(constraint, m_x);
        }
    }
}

scalar_t solver_state_t::value_test(const tensor_size_t patience) const
{
    assert(m_history_df.size() == m_history_dx.size());

    auto ii = m_history_df.size();
    auto dd = std::numeric_limits<scalar_t>::max();
    for (size_t it = m_history_df.size(); it > 0U; --it)
    {
        const auto df = m_history_df[it - 1U];
        const auto dx = m_history_dx[it - 1U];

        if (df > 0.0)
        {
            dd = std::max(df, dx);
            ii = it - 1U;
            break;
        }
    }

    // no improvement ever recorded, stop if enough iterations have passed
    if (ii == m_history_df.size())
    {
        return m_history_df.size() >= static_cast<size_t>(patience) ? 0.0 : dd;
    }

    // convergence criterion is the improvement in the recent iterations
    else if (ii + static_cast<size_t>(patience) >= m_history_df.size())
    {
        return dd;
    }

    // no improvement in the recent iterations, stop with potential convergence status
    else
    {
        return 0.0;
    }
}

scalar_t solver_state_t::gradient_test() const
{
    return m_gx.lpNorm<Eigen::Infinity>() / std::max(scalar_t(1), std::fabs(m_fx));
}

scalar_t solver_state_t::constraint_test() const
{
    scalar_t test = 0;
    if (m_ceq.size() > 0)
    {
        test += m_ceq.lpNorm<Eigen::Infinity>();
    }
    if (m_cineq.size() > 0)
    {
        test += m_cineq.array().max(0.0).matrix().lpNorm<Eigen::Infinity>();
    }
    return test;
}

bool solver_state_t::valid() const
{
    return std::isfinite(m_fx) && m_gx.array().isFinite().all() && m_ceq.array().isFinite().all() &&
           m_cineq.array().isFinite().all();
}

bool solver_state_t::has_armijo(const solver_state_t& origin, const vector_t& descent, const scalar_t step_size,
                                const scalar_t c1) const
{
    assert(c1 > 0 && c1 < 1);
    return m_fx <= origin.fx() + step_size * c1 * origin.dg(descent);
}

bool solver_state_t::has_approx_armijo(const solver_state_t& origin, const scalar_t epsilon) const
{
    return m_fx <= origin.fx() + epsilon;
}

bool solver_state_t::has_wolfe(const solver_state_t& origin, const vector_t& descent, const scalar_t c2) const
{
    assert(c2 > 0 && c2 < 1);
    return dg(descent) >= c2 * origin.dg(descent);
}

bool solver_state_t::has_strong_wolfe(const solver_state_t& origin, const vector_t& descent, const scalar_t c2) const
{
    assert(c2 > 0 && c2 < 1);
    return std::fabs(dg(descent)) <= c2 * std::fabs(origin.dg(descent));
}

bool solver_state_t::has_approx_wolfe(const solver_state_t& origin, const vector_t& descent, const scalar_t c1,
                                      const scalar_t c2) const
{
    assert(0 < c1 && c1 < scalar_t(0.5) && c1 < c2 && c2 < 1);
    return (2.0 * c1 - 1.0) * origin.dg(descent) >= dg(descent) && dg(descent) >= c2 * origin.dg(descent);
}

void solver_state_t::status(const solver_status status)
{
    m_status = status;
}

template <>
enum_map_t<solver_status> nano::enum_string<solver_status>()
{
    return {
        {solver_status::converged, "converged"},
        {solver_status::max_iters, "max_iters"},
        {   solver_status::failed,    "failed"},
        {  solver_status::stopped,   "stopped"}
    };
}

bool nano::operator<(const solver_state_t& lhs, const solver_state_t& rhs)
{
    return (std::isfinite(lhs.fx()) ? lhs.fx() : std::numeric_limits<scalar_t>::max()) <
           (std::isfinite(rhs.fx()) ? rhs.fx() : std::numeric_limits<scalar_t>::max());
}

std::ostream& nano::operator<<(std::ostream& stream, const solver_state_t& state)
{
    stream << "calls=" << state.fcalls() << "|" << state.gcalls();
    stream << ",f=" << state.fx() << ",g=" << state.gradient_test();
    if (state.ceq().size() + state.cineq().size() > 0)
    {
        stream << ",c=" << state.constraint_test();
    }
    return stream << "[" << state.status() << "]";
}

bool nano::converged(const solver_state_t& bstate, const solver_state_t& cstate, const scalar_t epsilon)
{
    const auto dx = (cstate.x() - bstate.x()).lpNorm<Eigen::Infinity>();

    return cstate.constraint_test() < epsilon && (dx < epsilon * std::max(1.0, bstate.x().lpNorm<Eigen::Infinity>()));
}
