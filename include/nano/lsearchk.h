#pragma once

#include <nano/configurable.h>
#include <nano/factory.h>
#include <nano/logger.h>
#include <nano/solver/lstep.h>

namespace nano
{
class lsearchk_t;
using rlsearchk_t = std::unique_ptr<lsearchk_t>;

///
/// \brief the objective type of the line-search procedure.
///
enum class lsearch_type : uint8_t
{
    none,               ///<
    armijo,             ///< sufficient decrease of the function value (Armijo)
    wolfe,              ///< armijo + decrease of the slope (Wolfe)
    strong_wolfe,       ///< armijo + small slow (Wolfe)
    wolfe_approx_wolfe, ///< armijo + wolfe or approximated armijo + wolfe (see CG_DESCENT)
};

template <>
inline enum_map_t<lsearch_type> enum_string<lsearch_type>()
{
    return {
        {              lsearch_type::none,                          "N/A"},
        {            lsearch_type::armijo,                       "Armijo"},
        {             lsearch_type::wolfe,                        "Wolfe"},
        {      lsearch_type::strong_wolfe,                 "strong Wolfe"},
        {lsearch_type::wolfe_approx_wolfe, "Wolfe or approximative Wolfe"},
    };
}

///
/// \brief compute the step size along the given descent direction starting from the initial guess `t0`.
///
/// NB: the returned step size is positive and guaranteed to decrease the function value (if no failure).
///
class NANO_PUBLIC lsearchk_t : public typed_t, public configurable_t, public clonable_t<lsearchk_t>
{
public:
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
    result_t get(solver_state_t&, const vector_t& descent, scalar_t initial_step_size, const logger_t&) const;

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
    bool update(solver_state_t&, const solver_state_t&, const vector_t&, scalar_t, const logger_t&) const;

    virtual result_t do_get(const solver_state_t&, const vector_t&, scalar_t, solver_state_t&,
                            const logger_t&) const = 0;

private:
    // attributes
    lsearch_type m_type{lsearch_type::none}; ///<
};
} // namespace nano
