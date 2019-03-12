#pragma once

#include <nano/json.h>
#include <nano/factory.h>
#include <nano/function.h>

namespace nano
{
    class solver_t;
    class lsearch_init_t;
    class lsearch_strategy_t;

    using solver_factory_t = factory_t<solver_t>;
    using lsearch_init_factory_t = factory_t<lsearch_init_t>;
    using lsearch_strategy_factory_t = factory_t<lsearch_strategy_t>;

    using rsolver_t = solver_factory_t::trobject;
    using rlsearch_strategy_t = lsearch_strategy_factory_t::trobject;

    ///
    /// \brief returns all registered solvers and line-search algorithms.
    ///
    NANO_PUBLIC solver_factory_t& get_solvers();
    NANO_PUBLIC lsearch_init_factory_t& get_lsearch_inits();
    NANO_PUBLIC lsearch_strategy_factory_t& get_lsearch_strategies();

    class solver_state_t;
    using ref_solver_state_t = std::reference_wrapper<const solver_state_t>;

    ///
    /// \brief optimization state described by:
    ///     current point (x),
    ///     function value (f),
    ///     gradient (g),
    ///     descent direction (d),
    ///     line-search step (t).
    ///
    class solver_state_t
    {
    public:

        enum class status
        {
            converged,      ///< convergence criterion reached
            max_iters,      ///< maximum number of iterations reached without convergence (default)
            failed,         ///< optimization failed (e.g. line-search failed)
            stopped         ///< user requested stop
        };

        ///
        /// \brief default constructor
        ///
        solver_state_t() = default;

        ///
        /// \brief constructor
        ///
        template <typename tvector>
        solver_state_t(const function_t& ffunction, tvector x0) :
            function(&ffunction),
            x(std::move(x0)),
            g(vector_t::Zero(x.size())),
            d(vector_t::Zero(x.size()))
        {
            f = function->vgrad(x, &g);
        }

        ///
        /// \brief move to another point
        ///
        template <typename tvector>
        bool update(const tvector& xx)
        {
            assert(function);
            assert(x.size() == xx.size());
            assert(x.size() == function->size());
            x = xx;
            f = function->vgrad(x, &g);
            return static_cast<bool>(*this);
        }

        ///
        /// \brief line-search step along the descent direction of state0
        ///
        bool update(const solver_state_t& state0, const scalar_t tt)
        {
            t = tt;
            return update(state0.x + t * state0.d);
        }

        ///
        /// \brief check convergence: the gradient is relatively small
        ///
        bool converged(const scalar_t epsilon) const
        {
            return convergence_criterion() < epsilon;
        }

        ///
        /// \brief convergence criterion: relative gradient
        ///
        scalar_t convergence_criterion() const
        {
            return g.lpNorm<Eigen::Infinity>() / std::max(scalar_t(1), std::fabs(f));
        }

        ///
        /// \brief check divergence
        ///
        operator bool() const
        {
            return std::isfinite(t) && std::isfinite(f) && std::isfinite(convergence_criterion());
        }

        ///
        /// \brief compute the dot product between the gradient and the descent direction
        ///
        auto dg() const
        {
            return g.dot(d);
        }

        ///
        /// \brief check if the chosen direction is a descent direction
        ///
        auto has_descent() const
        {
            return dg() < 0;
        }

        ///
        /// \brief check if the current step satisfies the Armijo condition (sufficient decrease)
        ///
        bool has_armijo(const solver_state_t& state0, const scalar_t c1) const
        {
            assert(c1 > 0 && c1 < 1);
            return f <= state0.f + t * c1 * state0.dg();
        }

        ///
        /// \brief check if the current step satisfies the approximate Armijo condition (sufficient decrease)
        ///     see CG_DESCENT
        ///
        bool has_approx_armijo(const solver_state_t& state0, const scalar_t epsilon) const
        {
            assert(epsilon > 0);
            return f <= state0.f + epsilon;
        }

        ///
        /// \brief check if the current step satisfies the Wolfe condition (sufficient curvature)
        ///
        bool has_wolfe(const solver_state_t& state0, const scalar_t c2) const
        {
            assert(c2 > 0 && c2 < 1);
            return dg() >= c2 * state0.dg();
        }

        ///
        /// \brief check if the current step satisfies the strong Wolfe condition (sufficient curvature)
        ///
        bool has_strong_wolfe(const solver_state_t& state0, const scalar_t c2) const
        {
            assert(c2 > 0 && c2 < 1);
            return std::fabs(dg()) <= c2 * std::fabs(state0.dg());
        }

        ///
        /// \brief check if the current step satisfies the approximate Wolfe condition (sufficient curvature)
        ///     see CG_DESCENT
        ///
        bool has_approx_wolfe(const solver_state_t& state0, const scalar_t c1, const scalar_t c2) const
        {
            assert(0 < c1 && c1 < scalar_t(0.5) && c1 < c2 && c2 < 1);
            return (2 * c1 - 1) * state0.dg() >= dg() && dg() >= c2 * state0.dg();
        }

        // attributes
        const function_t*   function{nullptr};      ///<
        vector_t            x, g, d;                ///< parameter, gradient, descent direction
        scalar_t            f{0}, t{0};             ///< function value, step size
        status              m_status{status::max_iters};    ///< optimization status
        size_t              m_fcalls{0};            ///< #function value evaluations so far
        size_t              m_gcalls{0};            ///< #function gradient evaluations so far
        size_t              m_iterations{0};        ///< #optimization iterations so far
    };

    template <>
    inline enum_map_t<solver_state_t::status> enum_string<solver_state_t::status>()
    {
        return
        {
            { solver_state_t::status::converged,   "converged" },
            { solver_state_t::status::max_iters,   "max_iters" },
            { solver_state_t::status::failed,      "failed" },
            { solver_state_t::status::stopped,     "stopped" }
        };
    }

    inline bool operator<(const solver_state_t& one, const solver_state_t& two)
    {
        return  (std::isfinite(one.f) ? one.f : std::numeric_limits<scalar_t>::max()) <
                (std::isfinite(two.f) ? two.f : std::numeric_limits<scalar_t>::max());
    }

    inline std::ostream& operator<<(std::ostream& os, const solver_state_t::status status)
    {
        return os << to_string(status);
    }

    ///
    /// \brief line-search step function:
    ///     phi(t) = f(x + t * d), f - the function to minimize and d - the descent direction.
    ///
    struct lsearch_step_t
    {
        lsearch_step_t() = default;
        lsearch_step_t(const lsearch_step_t&) = default;
        lsearch_step_t(const solver_state_t& state) : t(state.t), f(state.f), g(state.dg()) {}
        lsearch_step_t(const scalar_t tt, const scalar_t ff, const scalar_t gg) : t(tt), f(ff), g(gg) {}

        lsearch_step_t& operator=(const lsearch_step_t&) = default;
        lsearch_step_t& operator=(const solver_state_t& state)
        {
            t = state.t, f = state.f, g = state.dg();
            return *this;
        }

        ///
        /// \brief cubic interpolation of two line-search steps.
        ///     fit cubic: q(x) = a*x^3 + b*x^2 + c*x + d
        ///         given: q(u) = fu, q'(u) = gu
        ///         given: q(v) = fv, q'(v) = gv
        ///     minimizer: solution of 3*a*x^2 + 2*b*x + c = 0
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
        ///
        static auto cubic(const lsearch_step_t& u, const lsearch_step_t& v)
        {
            const auto d1 = u.g + v.g - 3 * (u.f - v.f) / (u.t - v.t);
            const auto d2 = (v.t > u.t ? +1 : -1) * std::sqrt(d1 * d1 - u.g * v.g);
            return v.t - (v.t - u.t) * (v.g + d2 - d1) / (v.g - u.g + 2 * d2);
        }

        ///
        /// \brief quadratic interpolation of two line-search steps.
        ///     fit quadratic: q(x) = a*x^2 + b*x + c
        ///         given: q(u) = fu, q'(u) = gu
        ///         given: q(v) = fv
        ///     minimizer: -b/2a
        ///
        static auto quadratic(const lsearch_step_t& u, const lsearch_step_t& v, bool* convexity = nullptr)
        {
            const auto dt = u.t - v.t;
            const auto df = u.f - v.f;
            if (convexity)
            {
                *convexity = (u.g - df / dt) * dt > 0;
            }
            return u.t - u.g * dt * dt / (2 * (u.g * dt - df));
        }

        ///
        /// \brief secant interpolation of two line-search steps.
        ///     fit quadratic: q(x) = a*x^2 + b*x + c
        ///         given: q'(u) = gu
        ///         given: q'(v) = gv
        ///     minimizer: -b/2a
        ///
        static auto secant(const lsearch_step_t& u, const lsearch_step_t& v)
        {
            return (v.t * u.g - u.t * v.g) / (u.g - v.g);
        }

        ///
        /// \brief bisection interpolation of two line-search steps.
        ///
        static auto bisect(const lsearch_step_t& u, const lsearch_step_t& v)
        {
            return (u.t + v.t) / 2;
        }

        // attributes
        scalar_t t{0};  ///< line-search step
        scalar_t f{0};  ///< line-search function value
        scalar_t g{0};  ///< line-search gradient
    };

    ///
    /// \brief compute the initial step length of the line search procedure.
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
    ///
    class lsearch_init_t : public json_configurable_t
    {
    public:

        lsearch_init_t() = default;

        ///
        /// \brief returns the initial step length given the current state
        /// NB: may keep track of the previous states
        ///
        scalar_t get(const solver_state_t& state)
        {
            return get(state, m_iteration ++);
        }

    private:

        virtual scalar_t get(const solver_state_t&, const int iteration) = 0;

        // attributes
        int         m_iteration{0}; ///<
    };

    ///
    /// \brief compute the step length of the line search procedure.
    ///
    class lsearch_strategy_t : public json_configurable_t
    {
    public:

        ///
        /// logging operator: op(solver_state_t), called for each trial of the line-search length.
        ///
        using logger_t = std::function<void(const solver_state_t&)>;

        ///
        /// \brief
        ///
        virtual bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t&) = 0;

        ///
        /// \brief change parameters
        ///
        auto& c1(const scalar_t c1) { m_c1 = c1; return *this; }
        auto& c2(const scalar_t c2) { m_c2 = c2; return *this; }
        auto& logger(const logger_t& logger) { m_logger = logger; return *this; }
        auto& max_iterations(const int max_iterations) { m_max_iterations = max_iterations; return *this; }

        ///
        /// \brief access functions
        ///
        auto c1() const { return m_c1; }
        auto c2() const { return m_c2; }
        auto max_iterations() const { return m_max_iterations; }

        ///
        /// \brief minimum allowed line-search step
        ///
        static scalar_t stpmin()
        {
            return scalar_t(10) * std::numeric_limits<scalar_t>::epsilon();
        }

        ///
        /// \brief maximum allowed line-search step
        ///
        static scalar_t stpmax()
        {
            return scalar_t(1) / stpmin();
        }

    protected:

        ///
        /// \brief log the current line-search trial length (if the logger is provided)
        ///
        void log(const solver_state_t& state) const
        {
            if (m_logger)
            {
                m_logger(state);
            }
        }

    private:

        // attributes
        scalar_t    m_c1{static_cast<scalar_t>(1e-4)};      ///< sufficient decrease rate
        scalar_t    m_c2{static_cast<scalar_t>(0.1)};       ///< sufficient curvature
        int         m_max_iterations{40};                   ///< #maximum iterations
        logger_t    m_logger;                               ///<
    };

    ///
    /// \brief line-search algorithm.
    ///
    class lsearch_t
    {
    public:

        ///
        /// \brief initial step length strategy
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
        ///     see CG_DESCENT papers
        ///
        enum class initializer
        {
            unit,                   ///< 1.0 (useful for quasi-Newton and Newton methods)
            linear,                 ///< consistent first-order change in the function
            quadratic,              ///< quadratic local interpolation (previous & current position)
            cgdescent,              ///< CG_DESCENT
        };

        ///
        /// \brief line-search strategy
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
        ///     see CG_DESCENT papers
        ///     see "Line Search Algorithms with Guaranteed Sufficient Decrease", J. J. More & D. J. Thuente
        ///     see "A view of line-searches", C. Lemarechal
        ///
        enum class strategy
        {
            backtrack,              ///< backtracking with sufficient decrease (Armijo)
            cgdescent,              ///< CG_DESCENT (regular and approximation Wolfe)
            lemarechal,             ///< Lemarechal (regular Wolfe)
            morethuente,            ///< More & Thunte (strong Wolfe)
            nocedalwright,          ///< Nocedal & Wright (strong Wolfe)
        };

        ///
        /// \brief constructor
        ///
        lsearch_t(const initializer, const strategy, const scalar_t c1, const scalar_t c2);

        ///
        /// \brief update the current state
        ///
        bool operator()(solver_state_t& state);

    private:

        // attributes
        std::unique_ptr<lsearch_init_t>         m_initializer;  ///<
        std::unique_ptr<lsearch_strategy_t>     m_strategy;     ///<
    };

    template <>
    inline enum_map_t<lsearch_t::initializer> enum_string<lsearch_t::initializer>()
    {
        return
        {
            { lsearch_t::initializer::unit,         "unit" },
            { lsearch_t::initializer::linear,       "linear" },
            { lsearch_t::initializer::quadratic,    "quadratic" },
            { lsearch_t::initializer::cgdescent,    "cgdescent" }
        };
    }

    template <>
    inline enum_map_t<lsearch_t::strategy> enum_string<lsearch_t::strategy>()
    {
        return
        {
            { lsearch_t::strategy::backtrack,       "backtrack" },
            { lsearch_t::strategy::cgdescent,       "cgdescent" },
            { lsearch_t::strategy::lemarechal,      "lemarechal" },
            { lsearch_t::strategy::morethuente,     "morethuente" },
            { lsearch_t::strategy::nocedalwright,   "nocedalwright" }
        };
    }

    ///
    /// \brief wrapper to keep track of the number of function value and gradient calls.
    ///
    class solver_function_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit solver_function_t(const function_t& function) :
            function_t(function),
            m_function(function)
        {
        }

        ///
        /// \brief compute function value (and gradient if provided)
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override
        {
            m_fcalls += 1;
            m_gcalls += gx ? 1 : 0;
            return m_function.vgrad(x, gx);
        }

        ///
        /// \brief number of function evaluation calls
        ///
        size_t fcalls() const { return m_fcalls; }

        ///
        /// \brief number of function gradient calls
        ///
        size_t gcalls() const { return m_gcalls; }

    private:

        // attributes
        const function_t&   m_function;         ///<
        mutable size_t      m_fcalls{0};        ///< #function value evaluations
        mutable size_t      m_gcalls{0};        ///< #function gradient evaluations
    };

    ///
    /// \brief generic (batch) optimization algorithm typically using an adaptive line-search method.
    ///
    class NANO_PUBLIC solver_t : public json_configurable_t
    {
    public:

        ///
        /// logging operator: op(state), returns false if the optimization should stop
        ///
        using logger_t = std::function<bool(const solver_state_t&)>;

        ///
        /// \brief minimize the given function starting from the initial point x0 until:
        ///     - convergence is achieved (critical point, possiblly a local/global minima) or
        ///     - the maximum number of iterations is reached or
        ///     - the user canceled the optimization (using the logging function) or
        ///     - the solver failed (e.g. line-search failed)
        ///
        solver_state_t minimize(const function_t& f, const vector_t& x0) const
        {
            assert(f.size() == x0.size());
            return minimize(solver_function_t(f), x0);
        }

        ///
        /// \brief change parameters
        ///
        auto& logger(const logger_t& logger) { m_logger = logger; return *this; }
        auto& epsilon(const scalar_t epsilon) { m_epsilon = epsilon; return *this; }
        auto& max_iterations(const int max_iterations) { m_max_iterations = max_iterations; return *this; }

        ///
        /// \brief access functions
        ///
        auto epsilon() const { return m_epsilon; }
        auto max_iterations() const { return m_max_iterations; }

    protected:

        ///
        /// \brief minimize the given function starting from the initial point x0
        ///
        virtual solver_state_t minimize(const solver_function_t&, const vector_t& x0) const = 0;

        ///
        /// \brief log the current optimization state (if the logger is provided)
        ///
        auto log(const solver_state_t& state) const
        {
            return !m_logger ? true : m_logger(state);
        }

        ///
        /// \brief check if the optimization is done (convergence or error) after an iteration
        ///
        bool done(const solver_function_t& function, solver_state_t& state, const bool iter_ok) const
        {
            state.m_fcalls = function.fcalls();
            state.m_gcalls = function.gcalls();

            const auto step_ok = iter_ok && state;
            const auto converged = state.converged(epsilon());

            if (converged || !step_ok)
            {
                // either converged or failed
                state.m_status = converged ?
                    solver_state_t::status::converged :
                    solver_state_t::status::failed;
                log(state);
                return true;
            }
            else if (!log(state))
            {
                // stopping was requested
                state.m_status = solver_state_t::status::stopped;
                return true;
            }

            // OK, go on with the optimization
            return false;
        }

    private:

        // attributes
        scalar_t    m_epsilon{1e-6};        ///< required precision (uper bounds the magnitude of the gradient)
        int         m_max_iterations{1000}; ///< maximum number of iterations
        logger_t    m_logger;               ///<
    };
}
