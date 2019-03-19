#pragma once

#include <nano/json.h>
#include <nano/factory.h>
#include <nano/lsearch/step.h>
#include <nano/solver/state.h>

namespace nano
{
    class lsearch_init_t;
    using lsearch_init_factory_t = factory_t<lsearch_init_t>;
    using rlsearch_init_t = lsearch_init_factory_t::trobject;

    ///
    /// \brief estimate the initial step length of the line-search procedure.
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
    ///
    class lsearch_init_t : public json_configurable_t
    {
    public:

        ///
        /// \brief constructor
        ///
        lsearch_init_t() = default;

        ///
        /// \brief returns the available implementations
        ///
        static lsearch_init_factory_t& all();

        ///
        /// \brief returns the initial step length given the current state
        /// NB: may keep track of the previous states
        ///
        virtual scalar_t get(const solver_state_t& state) = 0;
    };
}
