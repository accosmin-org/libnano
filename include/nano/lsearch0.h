#pragma once

#include <nano/configurable.h>
#include <nano/factory.h>
#include <nano/solver/lstep.h>

namespace nano
{
class lsearch0_t;
using rlsearch0_t = std::unique_ptr<lsearch0_t>;

///
/// \brief estimate the initial step size of the line-search procedure.
///
/// see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
/// see "Practical methods of optimization", Fletcher, chapter 2
///
class NANO_PUBLIC lsearch0_t : public configurable_t, public clonable_t<lsearch0_t>
{
public:
    ///
    /// logging operator: op(solver_state, proposed_line_search_step_size).
    ///
    using logger_t = std::function<void(const solver_state_t&, const scalar_t)>;

    ///
    /// \brief constructor
    ///
    explicit lsearch0_t(string_t id);

    ///
    /// \brief returns the available implementations.
    ///
    static factory_t<lsearch0_t>& all();

    ///
    /// \brief returns the initial step size given the current state.
    ///
    /// NB: may keep track of the initial step sizes computed for previous calls.
    ///
    virtual scalar_t get(const solver_state_t& state, const vector_t& descent, scalar_t last_step_size) = 0;

    ///
    /// \brief set the logging operator.
    ///
    void logger(const logger_t& logger) { m_logger = logger; }

protected:
    void log(const solver_state_t& state, const scalar_t step_size) const
    {
        if (m_logger)
        {
            m_logger(state, step_size);
        }
    }

private:
    // attributes
    logger_t m_logger; ///<
};
} // namespace nano
