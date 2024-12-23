#pragma once

#include <Eigen/Dense>
#include <nano/function/linear.h>
#include <nano/function/quadratic.h>
#include <solver/program/state.h>

namespace nano
{
struct program_t
{
    program_t(const linear_program_t& program, linear_constraints_t constraints)
        : program_t(&program, matrix_t{}, program.c(), std::move(constraints))
    {
    }

    program_t(const quadratic_program_t& program, linear_constraints_t constraints)
        : program_t(&program, program.Q(), program.c(), std::move(constraints))
    {
    }

    program_t(const function_t* function, matrix_t Q, vector_t c, linear_constraints_t constraints);

    tensor_size_t n() const { return m_c.size(); }

    tensor_size_t p() const { return m_A.rows(); }

    tensor_size_t m() const { return m_G.rows(); }

    bool feasible(const state_t& state) const;

    const matrix_t& Q() const
    {
        assert(m_Q.size() > 0);
        return m_Q;
    }

    template <class thessvar, class trdual, class trprim>
    const vector_t& solve(const thessvar& hessvar, const trdual& rdual, const trprim& rprim) const
    {
        const auto n = this->n();
        const auto p = this->p();

        // setup additional hessian components
        if (!m_Q.size())
        {
            m_lmat.block(0, 0, n, n) = -hessvar;
        }
        else
        {
            m_lmat.block(0, 0, n, n) = Q() - hessvar;
        }

        // setup residuals
        m_lvec.segment(0, n) = -rdual;
        m_lvec.segment(n, p) = -rprim;

        // solve the system
        m_ldlt.compute(m_lmat.matrix());
        m_lsol.vector() = m_ldlt.solve(m_lvec.vector());
        return m_lsol;
    }

    template <class tvector>
    void update(const tvector& x, const tvector& u, const tvector& v, const scalar_t miu, state_t& state) const
    {
        const auto m = this->m();
        const auto p = this->p();

        // objective
        if (!m_Q.size())
        {
            state.m_fx    = x.dot(m_c.vector());
            state.m_rdual = m_c;
        }
        else
        {
            state.m_fx    = 0.5 * x.dot(Q() * x) + x.dot(m_c.vector());
            state.m_rdual = Q() * x + m_c;
        }

        // surrogate duality gap
        if (m > 0)
        {
            state.m_eta = -u.dot(m_G * x - m_h);
        }

        // residual contributions of linear equality constraints
        if (p > 0)
        {
            state.m_rdual += m_A.transpose() * v;
            state.m_rprim = m_A * x - m_b;
        }

        // residual contributions of linear inequality constraints
        if (m > 0)
        {
            const auto sm = static_cast<scalar_t>(m);
            state.m_rdual += m_G.transpose() * u;
            state.m_rcent = -state.m_eta / (miu * sm) - u.array() * (m_G * x - m_h).array();
        }
    }

    using lin_solver_t = Eigen::LDLT<eigen_matrix_t<scalar_t>>;

    // attributes
    const function_t*    m_function{nullptr}; ///< original function to minimize
    matrix_t             m_Q;         ///< objective: 1/2 * x.dot(Q * x) + c.dot(x)
    vector_t             m_c;         ///<
    matrix_t             m_A;         ///< equality constraint: A * x = b
    vector_t             m_b;         ///<
    matrix_t             m_G;         ///< inequality constraint: Gx <= h
    vector_t             m_h;         ///<
    mutable lin_solver_t m_ldlt;      ///< buffers for the linear system of equations coupling (dx, dv)
    mutable matrix_t     m_lmat;      ///<
    mutable vector_t     m_lvec;      ///<
    mutable vector_t     m_lsol;      ///<
};
} // namespace nano
