#include <nano/core/sampling.h>
#include <nano/program/solver.h>
#include <nano/solver/gsample.h>
#include <nano/tensor/algorithm.h>

using namespace nano;

namespace
{
auto make_program(const tensor_size_t p)
{
    const auto positive = program::make_greater(p, 0.0);
    const auto weighted = program::make_equality(vector_t::constant(p, 1.0), 1.0);
    return program::make_quadratic(matrix_t::zero(p, p), vector_t::zero(p), positive, weighted);
}

struct sampler_t
{
    explicit sampler_t(const tensor_size_t n)
        : m_X(2 * n + 1, n)
        , m_G(2 * n + 1, n)
        , m_rng(make_rng())
    {
    }

    template <typename tmatrix>
    void descent(program::quadratic_program_t& program, const tmatrix& W, vector_t& g)
    {
        const auto G = m_G.slice(0, m_psize);
        program.m_Q  = G * W * G.transpose();

        const auto solution = m_solver.solve(program);
        assert(solution.m_status == solver_status::converged);
        g = G.transpose() * solution.m_x.vector();
        g = W * g;
    }

    // attributes
    matrix_t          m_X;        ///< buffer of sample points (p, n)
    matrix_t          m_G;        ///< buffer of sample gradients (p, n)
    tensor_size_t     m_psize{0}; ///< current number of samples
    rng_t             m_rng;
    program::solver_t m_solver;
};

struct perturbation_t
{
    perturbation_t(const tensor_size_t n, const scalar_t c)
        : m_zero(vector_t::zero(n))
        , m_ksi(n)
        , m_c(c)
        , m_rng(make_rng())
    {
    }

    const vector_t& generate(const solver_state_t& state, const vector_t& g)
    {
        const auto radius = std::max(m_c * state.gx().dot(g) / state.gx().lpNorm<2>(), epsilon0<scalar_t>());
        assert(std::isfinite(radius));
        assert(radius > 0.0);
        sample_from_ball(m_zero, radius, m_ksi, m_rng);
        return m_ksi;
    }

    // attributes
    vector_t m_zero;
    vector_t m_ksi;
    scalar_t m_c{1e-6};
    rng_t    m_rng;
};

struct linesearch_t
{
    linesearch_t(const tensor_size_t n, const configurable_t& configurable, const string_t& basename)
        : m_beta(configurable.parameter(basename + "lsearch_beta").value<scalar_t>())
        , m_gamma(configurable.parameter(basename + "lsearch_gamma").value<scalar_t>())
        , m_max_iters(configurable.parameter(basename + "lsearch_max_iters").value<tensor_size_t>())
        , m_perturbation(n, configurable.parameter(basename + "lsearch_perturb_c").value<scalar_t>())
    {
    }

    template <typename thessian>
    scalar_t step(vector_t& x, const vector_t& g, solver_state_t& state, const thessian& H)
    {
        const auto& function = state.function();

        const auto d  = g + m_perturbation.generate(state, g);
        const auto df = m_beta * g.dot(H * g);

        auto t = 1.0;
        if (auto fx = function.vgrad(x = state.x() - t * d); fx < state.fx() - t * df)
        {
            // doubling phase
            for (auto iters = 0; iters < m_max_iters; ++iters)
            {
                if (t /= m_gamma, fx = function.vgrad(x = state.x() - t * d); fx >= state.fx() - t * df)
                {
                    t *= m_gamma;
                    state.update(x = state.x() - t * d);
                    return t;
                }
            }
        }
        else
        {
            // bisection phase
            for (auto iters = 0; iters < m_max_iters; ++iters)
            {
                if (t *= m_gamma, fx = function.vgrad(x = state.x() - t * d); fx < state.fx() - t * df)
                {
                    state.update(x = state.x() - t * d);
                    return t;
                }
            }
        }

        return 0.0;
    }

    // attributes
    scalar_t       m_beta{1e-8};
    scalar_t       m_gamma{0.5};
    tensor_size_t  m_max_iters{50};
    perturbation_t m_perturbation;
};
} // namespace

struct gs::fixed_sampler_t final : public sampler_t
{
    explicit fixed_sampler_t(const tensor_size_t n)
        : sampler_t(n)
        , m_program(make_program(m_G.rows()))
    {
    }

    void sample(const solver_state_t& state, const scalar_t epsilon)
    {
        const auto m = m_G.rows() - 1;

        m_psize = 0;
        for (tensor_size_t i = 0; i < m; ++i, ++m_psize)
        {
            sample_from_ball(state.x(), epsilon, m_X.tensor(i), m_rng);
            assert((state.x() - m_X.tensor(i)).lpNorm<2>() <= epsilon + std::numeric_limits<scalar_t>::epsilon());
            state.function().vgrad(m_X.tensor(i), m_G.tensor(i));
        }

        m_X.tensor(m) = state.x();
        m_G.tensor(m) = state.gx();
        ++m_psize;

        assert(m_psize == m_X.rows());
        assert(m_psize == m_G.rows());
    }

    template <typename tmatrix>
    void descent(const tmatrix& W, vector_t& g)
    {
        sampler_t::descent(m_program, W, g);
    }

    // attributes
    program::quadratic_program_t m_program;
};

struct gs::adaptive_sampler_t final : public sampler_t
{
    explicit adaptive_sampler_t(const tensor_size_t n)
        : sampler_t(n)
    {
    }

    void sample(const solver_state_t& state, const scalar_t epsilon)
    {
        const auto p    = m_X.rows();
        const auto n    = m_X.cols();
        const auto phat = std::max(n / 10, tensor_size_t{1});

        // remove previously selected points outside the current ball
        const auto op1 = [&](const tensor_size_t i) { return (state.x() - m_X.tensor(i)).lpNorm<2>() > epsilon; };
        m_psize        = nano::remove_if(op1, m_X.slice(0, m_psize), m_G.slice(0, m_psize));

        // NB: to make sure at most `p` samples are used!
        const auto op2 = [&](const tensor_size_t i) { return i < (m_psize + 1 + phat - p); };
        m_psize        = nano::remove_if(op2, m_X.slice(0, m_psize), m_G.slice(0, m_psize));
        assert(m_psize + 1 + phat <= p);

        // current point (center of the ball)
        m_X.tensor(m_psize) = state.x();
        m_G.tensor(m_psize) = state.gx();
        ++m_psize;

        // new samples
        for (tensor_size_t i = 0; i < phat; ++i, ++m_psize)
        {
            assert(m_psize < p);
            sample_from_ball(state.x(), epsilon, m_X.tensor(m_psize), m_rng);
            state.function().vgrad(m_X.tensor(m_psize), m_G.tensor(m_psize));
        }

        for (tensor_size_t i = 0; i < m_psize; ++i)
        {
            assert((state.x() - m_X.tensor(i)).lpNorm<2>() <= epsilon + std::numeric_limits<scalar_t>::max());
        }
    }

    template <typename tmatrix>
    void descent(const tmatrix& W, vector_t& g)
    {
        auto program = make_program(m_psize);
        sampler_t::descent(program, W, g);
    }
};

class gs::identity_preconditioner_t
{
public:
    explicit identity_preconditioner_t(const tensor_size_t n)
        : m_W(matrix_t::identity(n, n))
        , m_H(matrix_t::identity(n, n))
    {
    }

    void update(const scalar_t) {}

    void update(const sampler_t&, const solver_state_t&, const scalar_t) {}

    const auto& W() const { return m_W; }

    const auto& H() const { return m_H; }

private:
    using identity_t = decltype(matrix_t::identity(3, 3));

    // attributes
    identity_t m_W;
    identity_t m_H;
};

class gs::lbfgs_preconditioner_t
{
public:
    explicit lbfgs_preconditioner_t(const tensor_size_t n)
        : m_W(n, n)
        , m_H(n, n)
    {
    }

    void update(const scalar_t alpha)
    {
        // initialization scalar of the inverse Hessian update
        const auto miu_min = 1e-2; // FIXME: make it a parameter
        const auto miu_max = 1e+3; // FIXME: make it a parameter
        if (alpha < 1.0)
        {
            m_miu = std::min(2.0 * m_miu, miu_max);
        }
        else
        {
            m_miu = std::max(0.5 * m_miu, miu_min);
        }
    }

    void update(const sampler_t& sampler, const solver_state_t& state, const scalar_t epsilon)
    {
        const auto n     = m_W.rows();
        const auto gamma = 0.1;   // FIXME: make it a parameter
        const auto sigma = 100.0; // FIXME: make it a parameter

        m_W = (1.0 / m_miu) * matrix_t::identity(n, n);
        m_H = m_miu * matrix_t::identity(n, n);

        for (tensor_size_t i = 0; i < sampler.m_psize; ++i)
        {
            const auto d  = sampler.m_X.tensor(i) - state.x();
            const auto y  = sampler.m_G.tensor(i) - state.gx();
            const auto dy = d.dot(y);

            assert(d.dot(d) <= epsilon + std::numeric_limits<scalar_t>::epsilon());

            if (dy >= gamma * epsilon && y.dot(y) <= sigma * epsilon)
            {
                m_Q = matrix_t::identity(n, n) - (y * d.transpose()) / dy;
                m_W = m_Q.transpose() * m_W;
                m_W = m_W * m_Q;
                m_W += (d * d.transpose()) / dy;

                m_H -= (m_H * d * d.transpose() * m_H) / (d.transpose() * m_H * d);
                m_H += (y * y.transpose()) / dy;
            }
        }

        assert(program::is_psd(m_W));
        assert(program::is_psd(m_H));

        assert((m_W * m_H - matrix_t::identity(n, n)).lpNorm<Eigen::Infinity>() < 1e-9);
        assert((m_H * m_W - matrix_t::identity(n, n)).lpNorm<Eigen::Infinity>() < 1e-9);
    }

    const auto& W() const { return m_W; }

    const auto& H() const { return m_H; }

private:
    // attributes
    matrix_t m_W; ///< H^-1
    matrix_t m_H; ///<
    matrix_t m_Q; ///< FIXME: is this still needed?!
    scalar_t m_miu{1.0};
};

struct gs::gs_type_id_t
{
    static auto str() { return "gs"; }
};

struct gs::gs_lbfgs_type_id_t
{
    static auto str() { return "gs-lbfgs"; }
};

struct gs::ags_type_id_t
{
    static auto str() { return "ags"; }
};

struct gs::ags_lbfgs_type_id_t
{
    static auto str() { return "ags-lbfgs"; }
};

template <typename tsampler, typename tpreconditioner, typename ttype_id>
base_solver_gs_t<tsampler, tpreconditioner, ttype_id>::base_solver_gs_t()
    : solver_t(ttype_id::str())
{
    type(solver_type::non_monotonic);

    const auto basename = scat("solver::", ttype_id::str(), "::");

    register_parameter(parameter_t::make_scalar(basename + "miu0", 0, LE, 1e-6, LT, 1e+6));
    register_parameter(parameter_t::make_scalar(basename + "epsilon0", 0, LT, 0.1, LT, 1e+6));
    register_parameter(parameter_t::make_scalar(basename + "theta_miu", 0, LT, 1.0, LE, 1));
    register_parameter(parameter_t::make_scalar(basename + "theta_epsilon", 0, LT, 0.1, LE, 1));

    register_parameter(parameter_t::make_scalar(basename + "lsearch_beta", 0, LE, 1e-8, LT, 1));
    register_parameter(parameter_t::make_scalar(basename + "lsearch_gamma", 0, LT, 0.5, LT, 1));
    register_parameter(parameter_t::make_scalar(basename + "lsearch_perturb_c", 0, LE, 1e-6, LT, 1));
    register_parameter(parameter_t::make_integer(basename + "lsearch_max_iters", 0, LT, 50, LE, 100));
}

template <typename tsampler, typename tpreconditioner, typename ttype_id>
rsolver_t base_solver_gs_t<tsampler, tpreconditioner, ttype_id>::clone() const
{
    return std::make_unique<base_solver_gs_t<tsampler, tpreconditioner, ttype_id>>(*this);
}

template <typename tsampler, typename tpreconditioner, typename ttype_id>
solver_state_t base_solver_gs_t<tsampler, tpreconditioner, ttype_id>::do_minimize(const function_t& function,
                                                                                  const vector_t&   x0) const
{
    const auto basename      = scat("solver::", ttype_id::str(), "::");
    const auto max_evals     = parameter("solver::max_evals").template value<tensor_size_t>();
    const auto epsilon       = parameter("solver::epsilon").template value<scalar_t>();
    const auto miu0          = parameter(basename + "miu0").template value<scalar_t>();
    const auto epsilon0      = parameter(basename + "epsilon0").template value<scalar_t>();
    const auto theta_miu     = parameter(basename + "theta_miu").template value<scalar_t>();
    const auto theta_epsilon = parameter(basename + "theta_epsilon").template value<scalar_t>();

    const auto n = function.size();

    auto x        = vector_t{n};
    auto g        = vector_t{n};
    auto miuk     = miu0;
    auto epsilonk = epsilon0;
    auto state    = solver_state_t{function, x0};
    auto sampler  = tsampler{n};
    auto precond  = tpreconditioner{n};
    auto lsearch  = linesearch_t{n, *this, basename};

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // sample gradients within the given radius
        sampler.sample(state, epsilonk);

        // update preconditioner
        precond.update(sampler, state, epsilonk);

        // solve the quadratic problem to find the stabilized gradient
        sampler.descent(precond.W(), g);

        // check convergence
        const auto iter_ok   = g.all_finite() && epsilonk > std::numeric_limits<scalar_t>::epsilon();
        const auto converged = state.gradient_test(g) < epsilon && epsilonk < epsilon;
        if (solver_t::done(state, iter_ok, converged))
        {
            break;
        }

        // too small gradient, reduce sampling radius (potentially convergence detected)
        else if (const auto gnorm2 = g.lpNorm<2>(); gnorm2 <= miuk)
        {
            precond.update(1.0);

            miuk *= theta_miu;
            epsilonk *= theta_epsilon;
        }

        // line-search step
        else
        {
            const auto alphak = lsearch.step(x, g, state, precond.H());
            precond.update(alphak);

            /*if (alphak < std::numeric_limits<scalar_t>::epsilon())
            {
                // NB: line-search failed, reduce the sampling radius - see (1)
                // epsilonk *= theta_epsilon;
            }*/
        }
    }

    state.update_calls();
    return state;
}

template class nano::base_solver_gs_t<gs::fixed_sampler_t, gs::identity_preconditioner_t, gs::gs_type_id_t>;
template class nano::base_solver_gs_t<gs::adaptive_sampler_t, gs::identity_preconditioner_t, gs::ags_type_id_t>;

template class nano::base_solver_gs_t<gs::fixed_sampler_t, gs::lbfgs_preconditioner_t, gs::gs_lbfgs_type_id_t>;
template class nano::base_solver_gs_t<gs::adaptive_sampler_t, gs::lbfgs_preconditioner_t, gs::ags_lbfgs_type_id_t>;
