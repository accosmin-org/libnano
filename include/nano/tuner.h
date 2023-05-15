#pragma once

#include <nano/configurable.h>
#include <nano/factory.h>
#include <nano/tuner/callback.h>
#include <nano/tuner/space.h>
#include <nano/tuner/step.h>

namespace nano
{
class tuner_t;
using rtuner_t = std::unique_ptr<tuner_t>;

///
/// \brief strategy to iteratively optimize hyper-parameters of machine learning models.
///
/// NB: a candidate combination of hyper-parameter values is usually evaluated using some error function
///     computed on the validation split.
///
class NANO_PUBLIC tuner_t : public configurable_t, public clonable_t<tuner_t>
{
public:
    ///
    /// \brief constructor
    ///
    explicit tuner_t(string_t id);

    ///
    /// \brief returns the available implementations.
    ///
    static factory_t<tuner_t>& all();

    ///
    /// \brief optimize the given hyper-parameters and returns all the evaluated steps.
    ///
    tuner_steps_t optimize(const param_spaces_t&, const tuner_callback_t&) const;

private:
    virtual void do_optimize(const param_spaces_t&, const tuner_callback_t&, tuner_steps_t& steps) const = 0;
};
} // namespace nano
