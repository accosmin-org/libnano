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
    return std::sqrt(m_rdual.dot(m_rdual) + m_rcent.dot(m_rcent) + m_rprim.dot(m_rprim));
}

void solver_state_t::update(const matrix_t& Q, const vector_t& c, const matrix_t& A, const vector_t& b,
                            const matrix_t& G, const vector_t& h)
{
    m_kkt = 0.0;

    // test 1
    if (G.size() > 0)
    {
        m_kkt = std::max(m_kkt, (G * m_x - h).array().max(0.0).matrix().lpNorm<Eigen::Infinity>());
    }

    // test 2
    if (A.size() > 0)
    {
        m_kkt = std::max(m_kkt, (A * m_x - b).lpNorm<Eigen::Infinity>());
    }

    // test 3
    if (G.size() > 0)
    {
        m_kkt = std::max(m_kkt, (-m_u.array()).max(0.0).matrix().lpNorm<Eigen::Infinity>());
    }

    // test 4
    if (G.size() > 0)
    {
        m_kkt = std::max(m_kkt, (m_u.array() * (G * m_x - h).array()).matrix().lpNorm<Eigen::Infinity>());
    }

    // test 5
    if (Q.size() > 0)
    {
        const auto lgrad = Q * m_x + c + A.transpose() * m_v + G.transpose() * m_u;
        m_kkt            = std::max(m_kkt, lgrad.lpNorm<Eigen::Infinity>());
    }
    else
    {
        const auto lgrad = c + A.transpose() * m_v + G.transpose() * m_u;
        m_kkt            = std::max(m_kkt, lgrad.lpNorm<Eigen::Infinity>());
    }
}

std::ostream& nano::program::operator<<(std::ostream& stream, const solver_state_t& state)
{
    return stream << "i=" << state.m_iters << ",fx=" << state.m_fx << ",eta=" << state.m_eta
                  << ",rdual=" << state.m_rdual.lpNorm<Eigen::Infinity>()
                  << ",rcent=" << state.m_rcent.lpNorm<Eigen::Infinity>()
                  << ",rprim=" << state.m_rprim.lpNorm<Eigen::Infinity>() << ",kkt=" << state.m_kkt
                  << ",rcond=" << state.m_ldlt_rcond << (state.m_ldlt_positive ? "(+)" : "(-)") << "[" << state.m_status
                  << "]";
}
