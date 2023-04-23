#pragma once

#include <nano/core/configurable.h>
#include <nano/core/factory.h>
#include <nano/solver/lstep.h>

namespace nano
{
class lsearchk_t;
using rlsearchk_t = std::unique_ptr<lsearchk_t>;

///
/// \brief the objective type of the line-search procedure.
///
enum class lsearch_type
{
    none,               ///<
    armijo,             ///< sufficient decrease of the function value (Armijo)
    wolfe,              ///< armijo + decrease of the slope (Wolfe)
    strong_wolfe,       ///< armijo + small slow (Wolfe)
    wolfe_approx_wolfe, ///< armijo + wolfe or approximated armijo + wolfe (see CG_DESCENT)
};

template <>
NANO_PUBLIC enum_map_t<lsearch_type> enum_string<lsearch_type>();

///
/// \brief compute the step size along the given descent direction starting from the initial guess `t0`.
///
/// NB: the returned step size is positive and guaranteed to decrease the function value (if no failure).
///
class NANO_PUBLIC lsearchk_t : public configurable_t, public clonable_t<lsearchk_t>
{
public:
    ///
    /// logging operator called for each trial of the line-search step size:
    ///     op(solver_state_at_0, solver_state_at_t, descent direction, step_size).
    ///
    using logger_t = std::function<void(const solver_state_t&, const solver_state_t&, const vector_t&, scalar_t)>;

    ///
    /// line-search result: <success flag, step size>.
    ///
    using result_t = std::tuple<bool, scalar_t>;

    ///
    /// \brief constructor
    ///
    explicit lsearchk_t(string_t id);

    ///
    /// \brief returns the available implementations.
    ///
    static factory_t<lsearchk_t>& all();

    ///
    /// \brief compute the step size starting from the given state and the initial estimate of the step size.
    ///
    result_t get(solver_state_t&, const vector_t& descent, scalar_t initial_step_size) const;

    ///
    /// \brief set the logging operator.
    ///
    void logger(const logger_t& logger);

    ///
    /// \brief minimum allowed line-search step.
    ///
    static scalar_t stpmin();

    ///
    /// \brief maximum allowed line-search step.
    ///
    static scalar_t stpmax();

    ///
    /// \brief returns the objective type optimized by the line-search implementation.
    ///
    lsearch_type type() const;

protected:
    void type(lsearch_type);
    bool update(solver_state_t&, const solver_state_t&, const vector_t&, scalar_t) const;

    virtual result_t do_get(const solver_state_t&, const vector_t&, scalar_t, solver_state_t&) const = 0;

private:
    // attributes
    logger_t     m_logger;                   ///<
    lsearch_type m_type{lsearch_type::none}; ///<
};
} // namespace nano
