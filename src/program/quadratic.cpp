#include <Eigen/Dense>
#include <nano/program/quadratic.h>

using namespace nano;
using namespace nano::program;

namespace
{
matrix_t make_Q(const vector_t& q)
{
    const auto n = static_cast<tensor_size_t>(std::sqrt(static_cast<double>(2 * q.size())));
    assert(2 * q.size() == n * (n + 1));

    auto Q = matrix_t(n, n);
    for (tensor_size_t row = 0, idx = 0; row < n; ++row)
    {
        for (tensor_size_t col = row; col < n; ++col, ++idx)
        {
            Q(row, col) = q(idx);
            Q(col, row) = q(idx); // NOLINT(readability-suspicious-call-argument)
        }
    }

    return Q;
}
} // namespace

quadratic_program_t::quadratic_program_t(matrix_t Q, vector_t c)
    : m_Q(std::move(Q))
    , m_c(std::move(c))
{
    assert(m_c.size() > 0);
    assert(m_Q.rows() == m_c.size());
    assert(m_Q.cols() == m_c.size());
}

quadratic_program_t::quadratic_program_t(const vector_t& Q_upper_triangular, vector_t c)
    : quadratic_program_t(make_Q(Q_upper_triangular), std::move(c))
{
}

bool quadratic_program_t::convex() const
{
    if (!m_Q.isApprox(m_Q.transpose()))
    {
        return false;
    }

    const auto ldlt = m_Q.selfadjointView<Eigen::Upper>().ldlt();
    return ldlt.info() != Eigen::NumericalIssue && ldlt.isPositive();
}

quadratic_program_t nano::program::operator&(const quadratic_program_t& program, const equality_t& eq)
{
    auto result = program;
    result.m_eq = result.m_eq & eq;
    assert(!result.m_eq || result.m_eq.m_A.cols() == result.m_c.size());
    return result;
}

quadratic_program_t nano::program::operator&(const quadratic_program_t& program, const inequality_t& ineq)
{
    auto result   = program;
    result.m_ineq = result.m_ineq & ineq;
    assert(!result.m_ineq || result.m_ineq.m_A.cols() == result.m_c.size());
    return result;
}
