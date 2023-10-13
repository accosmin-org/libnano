#include <Eigen/Dense>
#include <nano/program/constraint.h>

using namespace nano;
using namespace nano::program;

namespace
{
template <typename tconstraint>
tconstraint concat(const tconstraint& lhs, const tconstraint& rhs)
{
    if (!lhs)
    {
        return rhs;
    }
    else if (!rhs)
    {
        return lhs;
    }
    else
    {
        assert(lhs.m_A.cols() == rhs.m_A.cols());

        return {stack<scalar_t>(lhs.m_A.rows() + rhs.m_A.rows(), lhs.m_A.cols(), lhs.m_A, rhs.m_A),
                stack<scalar_t>(lhs.m_b.size() + rhs.m_b.size(), lhs.m_b, rhs.m_b)};
    }
}
} // namespace

constraint_t::constraint_t() = default;

constraint_t::constraint_t(matrix_t A, vector_t b)
    : m_A(std::move(A))
    , m_b(std::move(b))
{
    assert(m_A.rows() == m_b.size());
}

constraint_t::constraint_t(const vector_t& a, const scalar_t b)
    : constraint_t(map_matrix(a.data(), 1, a.size()), map_vector(&b, 1))
{
}

constraint_t::operator bool() const
{
    return m_A.size() > 0;
}

bool equality_t::feasible(const vector_t& x, const scalar_t epsilon) const
{
    assert(static_cast<bool>(*this));
    return deviation(x) < epsilon;
}

bool inequality_t::feasible(const vector_t& x, const scalar_t epsilon) const
{
    assert(static_cast<bool>(*this));
    return deviation(x) < epsilon;
}

std::optional<vector_t> inequality_t::make_strictly_feasible() const
{
    assert(static_cast<bool>(*this));

    static constexpr auto epsil0 = 0.1;
    static constexpr auto trials = 20;

    const auto decomp = (m_A.transpose() * m_A).ldlt();

    std::optional<vector_t> ret;

    auto x = vector_t{m_A.cols()};
    for (auto trial = 0; trial < trials; ++trial)
    {
        const auto y = std::pow(epsil0, static_cast<scalar_t>(trial));

        x = decomp.solve(m_A.transpose() * (m_b + vector_t::Constant(m_A.rows(), -y)));
        if ((m_A * x - m_b).maxCoeff() < 0.0)
        {
            ret = std::move(x);
            break;
        }
    }

    return ret;
}

equality_t nano::program::operator&(const equality_t& lhs, const equality_t& rhs)
{
    return ::concat(lhs, rhs);
}

inequality_t nano::program::operator&(const inequality_t& lhs, const inequality_t& rhs)
{
    return ::concat(lhs, rhs);
}

bool linear_constrained_t::feasible(const vector_t& x, const scalar_t epsilon) const
{
    return (!m_eq || m_eq.feasible(x, epsilon)) && (!m_ineq || m_ineq.feasible(x, epsilon));
}

bool linear_constrained_t::reduce()
{
    if (!m_eq)
    {
        return false;
    }

    auto& A = m_eq.m_A;
    auto& b = m_eq.m_b;

    const auto dd = A.transpose().fullPivLu();

    // independant linear constraints
    if (dd.rank() == A.rows())
    {
        return false;
    }

    // dependant linear constraints, use decomposition to formulate equivalent linear equality constraints
    const auto& P  = dd.permutationP();
    const auto& Q  = dd.permutationQ();
    const auto& LU = dd.matrixLU();

    const auto n = std::min(A.rows(), A.cols());
    const auto L = LU.leftCols(n).triangularView<Eigen::UnitLower>().toDenseMatrix();
    const auto U = LU.topRows(n).triangularView<Eigen::Upper>().toDenseMatrix();

    A = U.transpose().block(0, 0, dd.rank(), U.rows()) * L.transpose() * P;
    b = (Q.transpose() * b).segment(0, dd.rank());
    return true;
}
