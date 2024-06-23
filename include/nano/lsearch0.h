#pragma once

#include <nano/configurable.h>
#include <nano/factory.h>
#include <nano/loggable.h>
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
class NANO_PUBLIC lsearch0_t : public typed_t, public configurable_t, public loggable_t, public clonable_t<lsearch0_t>
{
public:
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
};
} // namespace nano
