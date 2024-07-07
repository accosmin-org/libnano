#pragma once

#include <nano/tuner.h>

namespace nano
{
///
/// \brief optimize hyper-parameters using local search in the neighborhood of the current optimum.
///
class NANO_PUBLIC local_search_tuner_t final : public tuner_t
{
public:
    ///
    /// \brief constructor
    ///
    local_search_tuner_t();

    ///
    /// \brief @see clonable_t
    ///
    rtuner_t clone() const override;

    ///
    /// \brief @see tuner_t
    ///
    void do_optimize(const param_spaces_t&, const tuner_callback_t&, const logger_t&, tuner_steps_t&) const override;
};
} // namespace nano
