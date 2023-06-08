#pragma once

#include <map>
#include <nano/configurable.h>
#include <nano/core/cmdline.h>

namespace nano
{
///
/// \brief RAII utility to keep track of the used parameters and
///     log all unused parameters at the end (e.g. typos, not matching to any solver).
///
class NANO_PUBLIC parameter_tracker_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit parameter_tracker_t(const cmdline_t::result_t&);

    ///
    /// \brief disable copying and moving.
    ///
    parameter_tracker_t(parameter_tracker_t&&)                 = delete;
    parameter_tracker_t(const parameter_tracker_t&)            = delete;
    parameter_tracker_t& operator=(parameter_tracker_t&&)      = delete;
    parameter_tracker_t& operator=(const parameter_tracker_t&) = delete;

    ///
    /// \brief configurable the given object and update the list of used parameters.
    ///
    void setup(configurable_t&);

    ///
    /// \brief destructor (log the unused parameters).
    ///
    ~parameter_tracker_t();

private:
    // attributes
    const cmdline_t::result_t& m_options;      ///<
    std::map<string_t, int>    m_params_usage; ///<
};
} // namespace nano
