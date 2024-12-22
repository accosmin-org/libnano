#include <Eigen/Dense>
#include <nano/function/linear.h>
#include <nano/function/quadratic.h>
#include <nano/function/util.h>
#include <solver/interior.h>

using namespace nano;

namespace
{
struct state_t
{
    state_t(const tensor_size_t n, const tensor_size_t n_ineqs, const tensor_size_t n_eqs)
        : m_x(vector_t::constant(n, nan))
        , m_u(vector_t::constant(n_ineqs, nan))
        , m_v(vector_t::constant(n_eqs, nan))
        , m_rdual(vector_t::constant(n, nan))
        , m_rcent(vector_t::constant(n_ineqs, nan))
        , m_rprim(vector_t::constant(n_eqs, nan))
    {
    }

    scalar_t residual() const { return std::sqrt(m_rdual.dot(m_rdual) + m_rcent.dot(m_rcent) + m_rprim.dot(m_rprim)); }

    void update(const matrix_t& Q, const vector_t& c, const matrix_t& A, const vector_t& b, const matrix_t& G,
                const vector_t& h)
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

    static constexpr auto max = std::numeric_limits<scalar_t>::max();
    static constexpr auto nan = std::numeric_limits<scalar_t>::quiet_NaN();

    // attributes
    int           m_iters{0};                         ///< number of iterations
    scalar_t      m_fx{nan};                          ///< objective
    vector_t      m_x;                                ///< solution (primal problem)
    vector_t      m_u;                                ///< Lagrange multipliers (inequality constraints)
    vector_t      m_v;                                ///< Lagrange multipliers (equality constraints)
    scalar_t      m_eta{nan};                         ///< surrogate duality gap
    vector_t      m_rdual;                            ///< dual residual
    vector_t      m_rcent;                            ///< central residual
    vector_t      m_rprim;                            ///< primal residual
    scalar_t      m_kkt{0};                           ///< KKT optimality test
    solver_status m_status{solver_status::max_iters}; ///< optimization status
    scalar_t      m_ldlt_rcond{0};                    ///< LDLT decomp: reciprocal condition number
    bool          m_ldlt_positive{false};             ///< LDLT decomp: positive semidefinite?, otherwise unstable
};

std::ostream& operator<<(std::ostream& stream, const state_t& state)
{
    return stream << "i=" << state.m_iters << ",fx=" << state.m_fx << ",eta=" << state.m_eta
                  << ",rdual=" << state.m_rdual.lpNorm<Eigen::Infinity>()
                  << ",rcent=" << state.m_rcent.lpNorm<Eigen::Infinity>()
                  << ",rprim=" << state.m_rprim.lpNorm<Eigen::Infinity>() << ",kkt=" << state.m_kkt
                  << ",rcond=" << state.m_ldlt_rcond << (state.m_ldlt_positive ? "(+)" : "(-)") << "[" << state.m_status
                  << "]";
}

template <class tprogram>
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

scalar_t normalize(matrix_t& A, vector_t& b, const scalar_t min_norm = 1e-3)
{
    const auto denom = std::max({min_norm, A.lpNorm<2>(), b.lpNorm<2>()});
    A.array() /= denom;
    b.array() /= denom;
    return denom;
}

struct reducer_t
{
    reducer_t() = default;

    reducer_t(matrix_t& A, vector_t& b) { reduce(A, b); }
};
} // namespace

struct ipm_solver_t::program_t
{
    explicit program_t(const linear_program_t& program, const linear_constraints_t& constraints)
        : program_t(matrix_t{}, program.c(), constraints.m_A, constraints.m_b, constraints.m_G, constraints.m_h)
    {
    }

    explicit program_t(const quadratic_program_t& program, const linear_constraints_t& constraints)
        : program_t(program.Q(), program.c(), constraints.m_A, constraints.m_b, constraints.m_G, constraints.m_h)
    {
    }

    program_t(matrix_t Q, vector_t c, matrix_t A, vector_t b, matrix_t G, vector_t h)
        : m_Q(std::move(Q))
        , m_c(std::move(c))
        , m_A(std::move(A))
        , m_b(std::move(b))
        , m_G(std::move(G))
        , m_h(std::move(h))
        , m_reducer(m_A, m_b)           // remove linear dependant linear constraints (if any)
        , m_mufx(::normalize(m_Q, m_c)) // normalize objective
        , m_lmat(n() + p(), n() + p())
        , m_lvec(n() + p())
        , m_lsol(n() + p())
    {
        // normalize constraints
        ::normalize(m_A, m_b);
        ::normalize(m_G, m_h);

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

    tensor_size_t n() const { return m_c.size(); }

    tensor_size_t p() const { return m_A.rows(); }

    tensor_size_t m() const { return m_G.rows(); }

    bool feasible(const solver_state_t& state) const
    {
        const auto& A = m_A;
        const auto& b = m_b;
        const auto& G = m_G;
        const auto& h = m_h;

        return (A.rows() == 0 || (A * state.m_x - b).lpNorm<2>() < epsilon2<scalar_t>()) &&
               (G.rows() == 0 || (G * state.m_x - h).maxCoeff() < epsilon2<scalar_t>());
    }

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
    void update(const tvector& x, const tvector& u, const tvector& v, const scalar_t miu, solver_state_t& state) const
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
        state.m_fx *= m_mufx; // NB: rescale the objective!

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
    matrix_t             m_Q;         ///< objective: 1/2 * x.dot(Q * x) + c.dot(x)
    vector_t             m_c;         ///<
    matrix_t             m_A;         ///< equality constraint: A * x = b
    vector_t             m_b;         ///<
    matrix_t             m_G;         ///< inequality constraint: Gx <= h
    vector_t             m_h;         ///<
    reducer_t            m_reducer;   ///<
    scalar_t             m_mufx{1.0}; ///< scaling coefficient of the objective
    mutable lin_solver_t m_ldlt;      ///< buffers for the linear system of equations coupling (dx, dv)
    mutable matrix_t     m_lmat;      ///<
    mutable vector_t     m_lvec;      ///<
    mutable vector_t     m_lsol;      ///<
};

ipm_solver_t::ipm_solver_t()
    : solver_t("ipm")
{
    register_parameter(parameter_t::make_scalar("solver::ipm::s0", 0.0, LT, 0.999, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::miu", 1.0, LT, 10.0, LE, 1e+6));
    register_parameter(parameter_t::make_scalar("solver::ipm::alpha", 0.0, LT, 1e-2, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::beta", 0.0, LT, 0.9, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::epsilon0", 0.0, LE, 1e-16, LE, 1e-3));
    register_parameter(parameter_t::make_integer("solver::ipm::max_iters", 10, LE, 300, LE, 1000));
    register_parameter(parameter_t::make_integer("solver::ipm::max_lsearch_iters", 10, LE, 50, LE, 1000));
}

rsolver_t ipm_solver_t::clone() const
{
    return std::make_unique<ipm_solver_t>(*this);
}

solver_state_t ipm_solver_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    if (const auto lconstraints = make_linear_constraints(function); !lconstraints)
    {
        critical0("interior point solver can only solve linearly-constrained functions!");
    }
    else if (const auto* const lprogram = dynamic_cast<const linear_program_t*>(&function); lprogram)
    {
        return do_minimize(program_t{*lprogram, *lconstraints}, x0, logger);
    }
    else if (const auto* const qprogram = dynamic_cast<const quadratic_program_t*>(&function); qprogram)
    {
        return do_minimize(program_t{*qprogram, *lconstraints}, x0, logger);
    }
    else
    {
        critical0("interior point solver can only solve linear and quadratic programs!");
    }
}

solver_state_t ipm_solver_t::do_minimize(const program_t& program, const vector_t& x0, const logger_t& logger) const
{
    if (program.m() > 0)
    {
        return do_minimize_with_inequality(program, x0, logger);
    }
    else
    {
        return do_minimize_without_inequality(program, x0, logger);
    }
}

solver_state_t ipm_solver_t::do_mimimize_with_inequality(const program_t& program, const vector_t& x0,
                                                         const logger_t& logger) const
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
        logger.info("[program]: ", state, ".\n");
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
            done(program, state, epsilon, logger);
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
            done(program, state, epsilon, logger);
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
            done(program, state, epsilon, logger);
            break;
        }

        // update state
        state.m_x += s * dx;
        state.m_u += s * du;
        state.m_v += s * dv;
        state.update(program.m_Q, program.m_c, program.m_A, program.m_b, program.m_G, program.m_h);

        const auto curr_eta   = state.m_eta;
        const auto curr_rdual = state.m_rdual.lpNorm<2>();
        const auto curr_rprim = state.m_rprim.lpNorm<2>();

        // check stopping criteria
        if (!std::isfinite(curr_eta) || !std::isfinite(curr_rdual) || !std::isfinite(curr_rprim))
        {
            // numerical instabilities
            state.m_status = solver_status::failed;
            logger.info("[program]: ", state, ",feasible=", program.feasible(state), ".\n");
            break;
        }
        else if (std::max({prev_eta - curr_eta, prev_rdual - curr_rdual, prev_rprim - curr_rprim}) < epsilon0)
        {
            // very precise convergence detected, check global convergence criterion!
            done(program, state, epsilon, logger);
            break;
        }
        else
        {
            logger.info("[program]: ", state, ",feasible=", program.feasible(state), ".\n");
        }
    }

    return state;
}

solver_state_t ipm_solver_t::do_minimize_without_inequality(const program_t& program, const vector_t& x0,
                                                            const logger_t& logger) const
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
    state.update(program.m_Q, program.m_c, program.m_A, program.m_b, program.m_G, program.m_h);

    const auto valid = std::isfinite(state.residual());
    const auto aprox = (program.m_lmat * program.m_lsol).isApprox(program.m_lvec.vector(), epsilon2<scalar_t>());
    state.m_status =
        (valid && aprox) ? solver_status::converged : (!valid ? solver_status::failed : solver_status::unfeasible);

    logger.info("[program]: ", state, ".\n");
    return state;
}

void solver_t::done(const program_t& program, solver_state_t& state, const scalar_t epsilon, const logger_t& logger)
{
    const auto feasible = program.feasible(state);

    if (feasible && std::max({state.m_eta, state.m_rdual.lpNorm<2>(), state.m_rprim.lpNorm<2>()}) < epsilon)
    {
        state.m_status = solver_status::converged;
    }
    else
    {
        // FIXME: this is an heuristic, to search for a theoretically sound method to detect unboundness and
        // unfeasibility!
        state.m_status = feasible ? solver_status::unbounded : solver_status::unfeasible;
    }

    logger.info("[program]: ", state, ",feasible=", feasible, ".\n");
}
