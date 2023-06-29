#include <Eigen/Dense>
#include <nano/solver/linprog.h>

using namespace nano;

linear_program_t::linear_program_t(vector_t c, matrix_t A, vector_t b)
    : m_c(std::move(c))
    , m_A(std::move(A))
    , m_b(std::move(b))
{
    assert(m_c.size() > 0);
    assert(m_b.size() > 0);
    assert(m_A.cols() == m_c.size());
    assert(m_A.rows() == m_b.size());
}

bool linear_program_t::feasible(const vector_t& x, const scalar_t epsilon) const
{
    assert(x.size() == m_c.size());
    return (m_A * x - m_b).lpNorm<Eigen::Infinity>() < epsilon && x.minCoeff() >= 0.0;
}

std::tuple<vector_t, vector_t, vector_t> nano::make_starting_point(const linear_program_t& prog)
{
    const auto& c = prog.m_c;
    const auto& A = prog.m_A;
    const auto& b = prog.m_b;

    const matrix_t invA = (A * A.transpose()).inverse();

    vector_t       x = A.transpose() * invA * b;
    const vector_t l = invA * A * c;
    vector_t       s = c - A.transpose() * l;

    const auto delta_x = std::max(0.0, -1.5 * x.minCoeff());
    const auto delta_s = std::max(0.0, -1.5 * s.minCoeff());

    x.array() += delta_x;
    s.array() += delta_s;

    const auto delta_x_hat = 0.5 * x.dot(s) / s.sum();
    const auto delta_s_hat = 0.5 * x.dot(s) / x.sum();

    return std::make_tuple(x.array() + delta_x_hat, l, s.array() + delta_s_hat);
}
