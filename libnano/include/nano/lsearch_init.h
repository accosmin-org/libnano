#pragma once

#include <nano/json.h>
#include <nano/factory.h>
#include <nano/solver_state.h>
#include <nano/lsearch_step.h>

namespace nano
{
    class lsearch_init_t;
    using lsearch_init_factory_t = factory_t<lsearch_init_t>;
    using rlsearch_init_t = lsearch_init_factory_t::trobject;

    ///
    /// \brief returns the registered line-search algorithms.
    ///
    NANO_PUBLIC lsearch_init_factory_t& get_lsearch_inits();

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
}
