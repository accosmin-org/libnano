#include <nano/function/constraint.h>
#include <nano/solver/state.h>

using namespace nano;

solver_state_t::dual_t::dual_t(const tensor_size_t n, const tensor_size_t nu, const tensor_size_t nv)
    : m_u(vector_t::zero(nu))
    , m_v(vector_t::zero(nv))
    , m_du(vector_t::zero(nu))
    , m_dv(vector_t::zero(nv))
    , m_gL(vector_t::zero(n))
{
}

solver_state_t::solver_state_t(const function_t& function, vector_t x0)
    : m_function(function)
    , m_x(std::move(x0))
    , m_gx(vector_t::zero(m_x.size()))
    , m_fx(m_function(m_x, m_gx))
    , m_dual(m_x.size(), m_function.n_inequalities(), m_function.n_equalities())
    , m_track(m_x, m_fx)
{
    assert(m_x.size() == m_function.size());

    update_calls();
    update_duals();
}

solver_state_t::solver_state_t(const solver_state_t& other) = default;

solver_state_t& solver_state_t::operator=(const solver_state_t& other)
{
    if (this != &other && &m_function == &other.m_function)
    {
        m_x      = other.m_x;
        m_gx     = other.m_gx;
        m_fx     = other.m_fx;
        m_dual   = other.m_dual;
        m_status = other.m_status;
        m_fcalls = other.m_fcalls;
        m_gcalls = other.m_gcalls;
        m_track  = other.m_track;
    }
    return *this;
}

bool solver_state_t::update(const vector_cmap_t x, const vector_cmap_t gx, const scalar_t fx, const vector_cmap_t u,
                            const vector_cmap_t v)
{
    assert(x.size() == m_x.size());
    assert(gx.size() == m_x.size());
    assert(u.size() == 0 || u.size() == m_dual.m_u.size());
    assert(v.size() == 0 || v.size() == m_dual.m_v.size());

    m_x  = x;
    m_gx = gx;
    m_fx = fx;

    if (u.size() == m_dual.m_u.size())
    {
        m_dual.m_u = u;
    }
    if (v.size() == m_dual.m_v.size())
    {
        m_dual.m_v = v;
    }

    update_calls();
    update_duals();
    return valid();
}

bool solver_state_t::update_if_better(const vector_t& x, const vector_t& gx, const scalar_t fx)
{
    update_calls();

    if (std::isfinite(fx))
    {
        const auto better = m_fx > fx;
        if (better)
        {
            m_x  = x;
            m_fx = fx;
            m_gx = gx;
            update_duals();
        }
        return better;
    }
    else
    {
        return false;
    }
}

bool solver_state_t::update_if_better(const vector_t& x, const scalar_t fx)
{
    return update_if_better(x, m_gx, fx);
}

void solver_state_t::update_calls()
{
    m_fcalls = m_function.fcalls();
    m_gcalls = m_function.gcalls();
    m_hcalls = m_function.hcalls();
}

void solver_state_t::update_history()
{
    m_track.update(m_x, m_fx);
}

void solver_state_t::update_duals()
{
    auto eq   = 0;
    auto ineq = 0;
    auto gc   = vector_t{m_x.size()};

    m_dual.m_gL = m_gx;
    for (const auto& constraint : m_function.constraints())
    {
        if (::nano::is_equality(constraint))
        {
            m_dual.m_dv(eq) = ::eval(constraint, m_x, gc);
            m_dual.m_gL += m_dual.m_v(eq) * gc;
            ++eq;
        }
        else
        {
            m_dual.m_du(ineq) = ::eval(constraint, m_x, gc);
            m_dual.m_gL += m_dual.m_u(ineq) * gc;
            ++ineq;
        }
    }

    m_dual.m_kkt1 = m_dual.m_du.array().max(0.0).matrix().lpNorm<Eigen::Infinity>();
    m_dual.m_kkt2 = m_dual.m_dv.lpNorm<Eigen::Infinity>();
    m_dual.m_kkt3 = (-m_dual.m_u.array()).max(0.0).matrix().lpNorm<Eigen::Infinity>();
    m_dual.m_kkt4 = (m_dual.m_u.array() * m_dual.m_du.array()).matrix().lpNorm<Eigen::Infinity>();
    m_dual.m_kkt5 = m_dual.m_gL.lpNorm<Eigen::Infinity>();

    m_dual.m_kkt1 /= 1.0 + std::fabs(m_fx);
    m_dual.m_kkt2 /= 1.0 + std::fabs(m_fx);
    m_dual.m_kkt3 /= 1.0 + std::fabs(m_fx);
    m_dual.m_kkt4 /= 1.0 + std::fabs(m_fx);
    m_dual.m_kkt5 /= 1.0 + std::fabs(m_fx);
}

scalar_t solver_state_t::value_test(const tensor_size_t patience) const
{
    if (m_function.constraints().empty())
    {
        return m_track.value_test_unconstrained(patience);
    }
    else
    {
        return m_track.value_test_constrained(patience);
    }
}

scalar_t solver_state_t::gradient_test() const
{
    return gradient_test(m_gx);
}

scalar_t solver_state_t::gradient_test(const vector_cmap_t gx) const
{
    return gx.lpNorm<Eigen::Infinity>() / std::max(1.0, std::fabs(m_fx));
}

bool solver_state_t::valid() const
{
    return std::isfinite(m_fx) && m_x.all_finite() && m_gx.all_finite() && m_dual.m_u.all_finite() &&
           m_dual.m_v.all_finite() && m_dual.m_du.all_finite() && m_dual.m_dv.all_finite() && m_dual.m_gL.all_finite();
}

bool solver_state_t::has_armijo(const solver_state_t& origin, const vector_t& descent, const scalar_t step_size,
                                const scalar_t c1) const
{
    assert(c1 > 0.0 && c1 < 1.0);
    assert((origin.x() + step_size * descent - x()).lpNorm<Eigen::Infinity>() < epsilon1<scalar_t>());

    return m_fx <= origin.fx() + step_size * c1 * origin.dg(descent);
}

bool solver_state_t::has_approx_armijo(const solver_state_t& origin, const scalar_t epsilon) const
{
    return m_fx <= origin.fx() + epsilon;
}

bool solver_state_t::has_wolfe(const solver_state_t& origin, const vector_t& descent, const scalar_t c2) const
{
    assert(c2 > 0.0 && c2 < 1.0);

    return dg(descent) >= c2 * origin.dg(descent);
}

bool solver_state_t::has_strong_wolfe(const solver_state_t& origin, const vector_t& descent, const scalar_t c2) const
{
    assert(c2 > 0.0 && c2 < 1.0);

    return std::fabs(dg(descent)) <= c2 * std::fabs(origin.dg(descent));
}

bool solver_state_t::has_approx_wolfe(const solver_state_t& origin, const vector_t& descent, const scalar_t c1,
                                      const scalar_t c2) const
{
    assert(0 < c1 && c1 < 0.5 && c1 < c2 && c2 < 1.0);

    return (2.0 * c1 - 1.0) * origin.dg(descent) >= dg(descent) && dg(descent) >= c2 * origin.dg(descent);
}

void solver_state_t::status(const solver_status status)
{
    m_status = status;
}

scalar_t solver_state_t::feasibility_test() const
{
    return std::max({kkt_optimality_test1(), kkt_optimality_test2()});
}

std::ostream& nano::operator<<(std::ostream& stream, const solver_state_t& state)
{
    stream << "calls=" << state.fcalls() << "|" << state.gcalls() << "|" << state.hcalls();
    stream << ",f=" << state.fx();
    if (state.function().smooth())
    {
        stream << ",g=" << state.gradient_test();
    }
    if (state.u().size() + state.v().size() > 0)
    {
        stream << ",kkt=(" << state.kkt_optimality_test1() << "," << state.kkt_optimality_test2() << ","
               << state.kkt_optimality_test3() << "," << state.kkt_optimality_test4() << ","
               << state.kkt_optimality_test5() << ")";
    }
    return stream << "[" << state.status() << "]";
}
