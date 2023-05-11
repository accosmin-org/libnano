#pragma once

#include <nano/solver/lsearch.h>

namespace nano
{
class solver_t;
using rsolver_t = std::unique_ptr<solver_t>;

///
/// \brief classifies numerical optimization algorithms (solvers).
///
enum class solver_type
{
    ///< descent is guaranteed at each step using line-search along a descent direction.
    ///< the constraints (if any) are ignored.
    ///< recommended for smooth unconstrained optimization problems.
    line_search,

    ///< descent is not guaranteed at each step.
    ///< the constraints (if any) and the line-search utilities are ignored.
    ///< recommended for non-smooth unconstrained optimization problems.
    non_monotonic,

    ///< handles the given constrains.
    ///< typically consists of solving a related unconstrained optimization in a loop.
    ///< recommended for constrained optimization problems.
    constrained,
};

///
/// \brief interface for numerical optimization algorithms.
///
/// NB: the resulting point for the unconstrained case (if enough iterations have been used) is either:
///     - the global minimum if the function is convex or
///     - a critical point (not necessarily a local minimum) otherwise.
///
class NANO_PUBLIC solver_t : public configurable_t, public clonable_t<solver_t>
{
public:
    ///
    /// \brief logging operator: op(state), returns false if the optimization should stop
    ///
    using logger_t = std::function<bool(const solver_state_t&)>;

    ///
    /// \brief constructor
    ///
    explicit solver_t(string_t id);

    ///
    /// \brief enable copying
    ///
    solver_t(const solver_t&);
    solver_t& operator=(const solver_t&) = delete;

    ///
    /// \brief enable moving
    ///
    solver_t(solver_t&&) noexcept   = default;
    solver_t& operator=(solver_t&&) = default;

    ///
    /// \brief destructor
    ///
    ~solver_t() override = default;

    ///
    /// \brief returns the available implementations.
    ///
    static factory_t<solver_t>& all();

    ///
    /// \brief minimize the given function starting from the initial point x0 until:
    ///     - convergence is achieved (critical point, possiblly a local/global minima) or
    ///     - the maximum number of iterations is reached or
    ///     - the user canceled the optimization (using the logging function) or
    ///     - the solver failed (e.g. line-search failed)
    ///
    solver_state_t minimize(const function_t&, const vector_t& x0) const;

    ///
    /// \brief set the logging callbacks.
    ///
    void logger(const logger_t& logger);
    void lsearch0_logger(const lsearch0_t::logger_t& logger);
    void lsearchk_logger(const lsearchk_t::logger_t& logger);

    ///
    /// \brief set the line-search initialization method.
    ///
    void lsearch0(const lsearch0_t&);
    void lsearch0(const string_t& id);

    ///
    /// \brief set the line-search strategy method.
    ///
    void lsearchk(const lsearchk_t&);
    void lsearchk(const string_t& id);

    ///
    /// \brief change the solver to be more precise by the given factor in the range (0, 1).
    ///
    void more_precise(scalar_t epsilon_factor);

    ///
    /// \brief returns the type of the optimization method.
    ///
    solver_type type() const;

    ///
    /// \brief return the line-search initialization method.
    ///
    const auto& lsearch0() const { return *m_lsearch0; }

    ///
    /// \brief return the the line-search strategy method.
    ///
    const auto& lsearchk() const { return *m_lsearchk; }

protected:
    void type(solver_type);
    bool log(const solver_state_t&) const;
    bool done(solver_state_t&, bool iter_ok, bool converged) const;

    lsearch_t make_lsearch() const;
    rsolver_t make_solver(const function_t&, scalar_t epsilon, tensor_size_t max_evals) const;

    virtual solver_state_t do_minimize(const function_t&, const vector_t& x0) const = 0;

private:
    // attributes
    logger_t    m_logger;                         ///<
    rlsearch0_t m_lsearch0;                       ///<
    rlsearchk_t m_lsearchk;                       ///<
    solver_type m_type{solver_type::line_search}; ///<
};
} // namespace nano

NANO_MAKE_ENUM3(solver_type, line_search, non_monotonic, constrained)
