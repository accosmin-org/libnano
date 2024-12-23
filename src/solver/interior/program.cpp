#include <nano/function/util.h>
#include <solver/interior/program.h>

using namespace nano;

program_t::program_t(const function_t* function, matrix_t Q, vector_t c, linear_constraints_t constraints)
    : m_function(function)
    , m_Q(std::move(Q))
    , m_c(std::move(c))
    , m_A(std::move(constraints.m_A))
    , m_b(std::move(constraints.m_b))
    , m_G(std::move(constraints.m_G))
    , m_h(std::move(constraints.m_h))
    , m_lmat(n() + p(), n() + p())
    , m_lvec(n() + p())
    , m_lsol(n() + p())
{
    // allocate buffers for the linear system of equations
    const auto n = this->n();
    const auto p = this->p();

    if (p > 0)
    {
        m_lmat.block(0, n, n, p) = m_A.transpose();
        m_lmat.block(n, 0, p, n) = m_A.matrix();
    }
    m_lmat.block(n, n, p, p).array() = 0.0;
}

bool program_t::feasible(const solver_state_t& state) const
{
    const auto& A = m_A;
    const auto& b = m_b;
    const auto& G = m_G;
    const auto& h = m_h;

    return (A.rows() == 0 || (A * state.m_x - b).lpNorm<2>() < epsilon2<scalar_t>()) &&
           (G.rows() == 0 || (G * state.m_x - h).maxCoeff() < epsilon2<scalar_t>());
}
