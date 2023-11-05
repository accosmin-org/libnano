#include <Eigen/Dense>
#include <nano/program/constrained.h>

using namespace nano;
using namespace nano::program;

bool linear_constrained_t::feasible(vector_cmap_t x, const scalar_t epsilon) const
{
    return (!m_eq.valid() || m_eq.feasible(x, epsilon)) && (!m_ineq.valid() || m_ineq.feasible(x, epsilon));
}

bool linear_constrained_t::reduce()
{
    if (!m_eq.valid())
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
    b = vector_t{(Q.transpose() * b).segment(0, dd.rank())};
    return true;
}

std::optional<vector_t> linear_constrained_t::make_strictly_feasible() const
{
    static constexpr auto epsil0 = 0.1;
    static constexpr auto trials = 20;

    std::optional<vector_t> ret;

    if (m_ineq.valid())
    {
        const auto& A      = m_ineq.m_A;
        const auto& b      = m_ineq.m_b;
        const auto  decomp = (A.transpose() * A).ldlt();

        auto x = vector_t{A.cols()};
        for (auto trial = 0; trial < trials; ++trial)
        {
            const auto y = std::pow(epsil0, static_cast<scalar_t>(trial));

            x.vector() = decomp.solve(A.transpose() * (b + vector_t::constant(A.rows(), -y)));
            if ((A * x.vector() - b).maxCoeff() < 0.0)
            {
                ret = std::move(x);
                break;
            }
        }
    }

    return ret;
}
