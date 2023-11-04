#include <Eigen/Dense>
#include <nano/program/solver.h>

using namespace nano;
using namespace nano::program;

namespace
{
template <typename tprogram>
vector_t make_x0(const tprogram& program)
{
    const auto x0 = program.make_strictly_feasible();
    if (!x0)
    {
        return vector_t::zero(program.m_c.size());
    }
    else
    {
        return x0.value();
    }
}

auto make_smax(const vector_t& u, const vector_t& du)
{
    assert(u.size() == du.size());

    auto smax = std::numeric_limits<scalar_t>::max();
    for (tensor_size_t i = 0, size = u.size(); i < size; ++i)
    {
        if (du(i) < 0.0)
        {
            smax = std::min(smax, -u(i) / du(i));
        }
    }

    return std::min(smax, 1.0);
}
} // namespace

struct solver_t::program_t
{
    using opt_matrix_t = std::optional<std::reference_wrapper<const matrix_t>>;

    explicit program_t(const linear_program_t& program)
        : program_t({}, program.m_c, program.m_eq.m_A, program.m_eq.m_b, program.m_ineq.m_A, program.m_ineq.m_b)
    {
    }

    explicit program_t(const quadratic_program_t& program)
        : program_t(std::cref(program.m_Q), program.m_c, program.m_eq.m_A, program.m_eq.m_b, program.m_ineq.m_A,
                    program.m_ineq.m_b)
    {
    }

    program_t(const opt_matrix_t& Q, const vector_t& c, const matrix_t& A, const vector_t& b, const matrix_t& G,
              const vector_t& h)
        : m_Q(Q)
        , m_c(c)
        , m_A(A)
        , m_b(b)
        , m_G(G)
        , m_h(h)
        , m_lmat(n() + p(), n() + p())
        , m_lvec(n() + p())
        , m_lsol(n() + p())
    {
        const auto n = this->n();
        const auto p = this->p();

        if (p > 0)
        {
            m_lmat.block(0, n, n, p) = m_A.transpose();
            m_lmat.block(n, 0, p, n) = m_A.matrix();
        }
        m_lmat.block(n, n, p, p).array() = 0.0;
    }

    tensor_size_t n() const { return m_c.size(); }

    tensor_size_t p() const { return m_A.rows(); }

    tensor_size_t m() const { return m_G.rows(); }

    const matrix_t& Q() const { return m_Q.value().get(); }

    template <typename thessvar, typename trdual, typename trprim>
    const vector_t& solve(const thessvar& hessvar, const trdual& rdual, const trprim& rprim) const
    {
        const auto n = this->n();
        const auto p = this->p();

        // setup additional hessian components
        if (m_Q)
        {
            m_lmat.block(0, 0, n, n) = Q() - hessvar;
        }
        else
        {
            m_lmat.block(0, 0, n, n) = -hessvar;
        }

        // setup residuals
        m_lvec.segment(0, n) = -rdual;
        m_lvec.segment(n, p) = -rprim;

        // solve the system
        m_ldlt.compute(m_lmat.matrix());
        m_lsol.vector() = m_ldlt.solve(m_lvec.vector());
        return m_lsol;
    }

    template <typename tvector>
    void update(const tvector& x, const tvector& u, const tvector& v, const scalar_t miu, solver_state_t& state) const
    {
        const auto m = this->m();
        const auto p = this->p();

        // objective
        if (!m_Q)
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
    opt_matrix_t         m_Q;    ///< objective: 1/2 * x.dot(Q * x) + c.dot(x)
    const vector_t&      m_c;    ///<
    const matrix_t&      m_A;    ///< equality constraint: A * x = b
    const vector_t&      m_b;    ///<
    const matrix_t&      m_G;    ///< inequality constraint: Gx <= h
    const vector_t&      m_h;    ///<
    mutable lin_solver_t m_ldlt; ///< buffers for the linear system of equations coupling (dx, dv)
    mutable matrix_t     m_lmat; ///<
    mutable vector_t     m_lvec; ///<
    mutable vector_t     m_lsol; ///<
};

solver_t::solver_t(logger_t logger)
    : m_logger(std::move(logger))
{
    register_parameter(parameter_t::make_scalar("solver::s0", 0.0, LT, 0.99, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::miu", 1.0, LT, 10.0, LE, 1e+6));
    register_parameter(parameter_t::make_scalar("solver::alpha", 0.0, LT, 0.01, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::beta", 0.0, LT, 0.5, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::epsilon", 0.0, LE, 1e-9, LE, 1e-3));
    register_parameter(parameter_t::make_scalar("solver::epsilon0", 0.0, LE, 1e-15, LE, 1e-3));
    register_parameter(parameter_t::make_integer("solver::max_iters", 10, LE, 100, LE, 1000));
    register_parameter(parameter_t::make_integer("solver::max_lsearch_iters", 10, LE, 30, LE, 1000));
}

solver_state_t solver_t::solve(const linear_program_t& program) const
{
    return !program.m_ineq.valid() ? solve_without_inequality(program_t{program})
                                   : solve_with_inequality(program_t{program}, make_x0(program));
}

solver_state_t solver_t::solve(const quadratic_program_t& program) const
{
    return !program.m_ineq.valid() ? solve_without_inequality(program_t{program})
                                   : solve_with_inequality(program_t{program}, make_x0(program));
}

solver_state_t solver_t::solve(const linear_program_t& program, const vector_t& x0) const
{
    return !program.m_ineq.valid() ? solve_without_inequality(program_t{program})
                                   : solve_with_inequality(program_t{program}, x0);
}

solver_state_t solver_t::solve(const quadratic_program_t& program, const vector_t& x0) const
{
    return !program.m_ineq.valid() ? solve_without_inequality(program_t{program})
                                   : solve_with_inequality(program_t{program}, x0);
}

solver_state_t solver_t::solve_with_inequality(const program_t& program, const vector_t& x0) const
{
    const auto s0                = parameter("solver::s0").value<scalar_t>();
    const auto miu               = parameter("solver::miu").value<scalar_t>();
    const auto alpha             = parameter("solver::alpha").value<scalar_t>();
    const auto beta              = parameter("solver::beta").value<scalar_t>();
    const auto epsilon           = parameter("solver::epsilon").value<scalar_t>();
    const auto epsilon0          = parameter("solver::epsilon0").value<scalar_t>();
    const auto max_iters         = parameter("solver::max_iters").value<tensor_size_t>();
    const auto max_lsearch_iters = parameter("solver::max_lsearch_iters").value<tensor_size_t>();

    const auto& G = program.m_G;
    const auto& h = program.m_h;
    const auto  n = program.n();
    const auto  p = program.p();
    const auto  m = program.m();

    auto state = solver_state_t{n, m, p};
    state.m_x  = x0;

    // the starting point must be strictly feasible wrt inequality constraints
    if (const auto mGxh = (G * x0 - h).maxCoeff(); mGxh >= 0.0)
    {
        state.m_status = solver_status::unfeasible;
        return state;
    }

    auto dx = vector_t{n};
    auto du = vector_t{m};
    auto dv = vector_t{p};

    // current solver state
    state.m_u = -1.0 / (G * x0 - h).array();
    state.m_v = vector_t::zero(p);

    // update residuals
    program.update(state.m_x, state.m_u, state.m_v, miu, state);

    // primal-dual interior-point solver...
    for (state.m_iters = 0; state.m_iters < max_iters; ++state.m_iters)
    {
        const auto prev_eta   = state.m_eta;
        const auto prev_rdual = state.m_rdual.lpNorm<2>();
        const auto prev_rprim = state.m_rprim.lpNorm<2>();

        // solve primal-dual linear system of equations to get (dx, du, dv)
        const auto Gxh = G * state.m_x - h;
        program.solve(G.transpose() * (state.m_u.array() / Gxh.array()).matrix().asDiagonal() * G.matrix(),
                      state.m_rdual + G.transpose() * (state.m_rcent.array() / Gxh.array()).matrix(), state.m_rprim);

        dx = program.m_lsol.segment(0, n);
        dv = program.m_lsol.segment(n, p);
        du = (state.m_rcent.array() - state.m_u.array() * (G * dx).array()) / Gxh.array();

        state.m_ldlt_rcond    = program.m_ldlt.rcond();
        state.m_ldlt_positive = program.m_ldlt.isPositive();

        // stop if the linear system of equations is not stable
        if (!std::isfinite(state.m_ldlt_rcond) || !dx.all_finite() || !dv.all_finite() || !du.all_finite())
        {
            done(state, epsilon);
            break;
        }

        // backtracking line-search: stage 1
        auto s    = s0 * make_smax(state.m_u, du);
        auto iter = tensor_size_t{0};
        for (iter = 0; iter < max_lsearch_iters; ++iter)
        {
            if ((G * (state.m_x + s * dx) - h).maxCoeff() < 0.0)
            {
                break;
            }
            else
            {
                s *= beta;
            }
        }
        if (iter == max_lsearch_iters)
        {
            done(state, epsilon);
            break;
        }

        // backtracking line-search: stage 2
        const auto r0 = state.residual();
        for (iter = 0; iter < max_lsearch_iters; ++iter)
        {
            program.update(state.m_x + s * dx, state.m_u + s * du, state.m_v + s * dv, miu, state);
            if (state.residual() <= (1.0 - alpha * s) * r0)
            {
                break;
            }
            else
            {
                s *= beta;
            }
        }
        if (iter == max_lsearch_iters)
        {
            // NB: revert to previous state if the residual didn't improve!
            if (state.residual() > r0)
            {
                program.update(state.m_x, state.m_u, state.m_v, miu, state);
            }
            done(state, epsilon);
            break;
        }

        // update state
        state.m_x += s * dx;
        state.m_u += s * du;
        state.m_v += s * dv;

        const auto curr_eta   = state.m_eta;
        const auto curr_rdual = state.m_rdual.lpNorm<2>();
        const auto curr_rprim = state.m_rprim.lpNorm<2>();

        // check stopping criteria
        if (!std::isfinite(curr_eta) || !std::isfinite(curr_rdual) || !std::isfinite(curr_rprim))
        {
            state.m_status = solver_status::failed;
            log(state);
            break;
        }
        else if (std::max({prev_eta - curr_eta, prev_rdual - curr_rdual, prev_rprim - curr_rprim}) < epsilon0)
        {
            done(state, epsilon);
            break;
        }
        else if (!log(state))
        {
            state.m_status = solver_status::stopped;
            break;
        }
    }

    return state;
}

solver_state_t solver_t::solve_without_inequality(const program_t& program) const
{
    const auto miu = parameter("solver::miu").value<scalar_t>();

    const auto& c = program.m_c;
    const auto& b = program.m_b;
    const auto  n = program.n();
    const auto  p = program.p();

    // NB: solve directly the KKT-based system of linear equations coupling (x, v)
    program.solve(matrix_t::zero(n, n), c, -b);

    // current solver state
    auto state  = solver_state_t{n, 0, p};
    state.m_x   = program.m_lsol.segment(0, n);
    state.m_v   = program.m_lsol.segment(n, p);
    state.m_eta = 0.0;
    program.update(state.m_x, state.m_u, state.m_v, miu, state);

    const auto epsil = std::sqrt(std::numeric_limits<scalar_t>::epsilon());
    const auto valid = std::isfinite(state.residual());
    const auto aprox = (program.m_lmat * program.m_lsol).isApprox(program.m_lvec.vector(), epsil);
    state.m_status   = (valid && aprox) ? solver_status::converged : solver_status::failed;

    log(state);
    return state;
}

void solver_t::done(solver_state_t& state, const scalar_t epsilon) const
{
    const auto curr_eta   = state.m_eta;
    const auto curr_rdual = state.m_rdual.lpNorm<Eigen::Infinity>();
    const auto curr_rprim = state.m_rprim.lpNorm<Eigen::Infinity>();

    const auto converged = std::max({curr_eta, curr_rdual, curr_rprim}) < epsilon;
    state.m_status       = converged ? solver_status::converged : solver_status::failed;
    log(state);
}

bool solver_t::log(const solver_state_t& state) const
{
    return !m_logger ? true : m_logger(state);
}
