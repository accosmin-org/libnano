#pragma once

#include <nano/program/solver.h>
#include <nano/solver/state.h>

using namespace nano;

namespace nano::gsample
{
struct sampler_t
{
    explicit sampler_t(tensor_size_t n);

    template <class tmatrix>
    void descent(program::quadratic_program_t& program, const tmatrix& W, vector_t& g, const logger_t& logger)
    {
        const auto G = m_G.slice(0, m_psize);
        program.m_Q  = G * W * G.transpose();

        const auto solution = m_solver.solve(program, logger);
        // FIXME: this cannot be guaranteed, better to show a warning!
        assert(solution.m_status == solver_status::converged);
        g = G.transpose() * solution.m_x.vector();
        g = W * g;
    }

    static program::quadratic_program_t make_program(tensor_size_t p);

    // attributes
    matrix_t          m_X;        ///< buffer of sample points (p, n)
    matrix_t          m_G;        ///< buffer of sample gradients (p, n)
    tensor_size_t     m_psize{0}; ///< current number of samples
    program::solver_t m_solver;
};

class fixed_sampler_t final : public sampler_t
{
public:
    explicit fixed_sampler_t(tensor_size_t n);

    static auto str() { return "gs"; }

    void sample(const solver_state_t& state, scalar_t epsilon);

    template <class tmatrix>
    void descent(const tmatrix& W, vector_t& g, const logger_t& logger)
    {
        sampler_t::descent(m_program, W, g, logger);
    }

private:
    // attributes
    program::quadratic_program_t m_program;
};

class adaptive_sampler_t final : public sampler_t
{
public:
    explicit adaptive_sampler_t(tensor_size_t n);

    static auto str() { return "ags"; }

    void sample(const solver_state_t& state, scalar_t epsilon);

    template <class tmatrix>
    void descent(const tmatrix& W, vector_t& g, const logger_t& logger)
    {
        auto program = make_program(m_psize);
        sampler_t::descent(program, W, g, logger);
    }
};
} // namespace nano::gsample
