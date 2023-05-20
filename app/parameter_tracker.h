#pragma once

#include <map>
#include <nano/configurable.h>
#include <nano/core/cmdline.h>
#include <nano/core/logger.h>

using namespace nano;

///
/// \brief RAII utility to keep track of the used parameters and
///     log all unused parameters at the end (e.g. typos, not matching to any solver).
///
class parameter_tracker_t
{
public:
    explicit parameter_tracker_t(const cmdline_t::result_t& options)
        : m_options(options)
    {
        for (const auto& [param_name, param_value] : m_options.m_xvalues)
        {
            m_params_usage[param_name] = 0;
        }
    }

    parameter_tracker_t(parameter_tracker_t&&)      = delete;
    parameter_tracker_t(const parameter_tracker_t&) = delete;

    parameter_tracker_t& operator=(parameter_tracker_t&&)      = delete;
    parameter_tracker_t& operator=(const parameter_tracker_t&) = delete;

    void setup(configurable_t& configurable)
    {
        for (const auto& [param_name, param_value] : m_options.m_xvalues)
        {
            if (configurable.parameter_if(param_name) != nullptr)
            {
                configurable.parameter(param_name) = param_value;
                m_params_usage[param_name]++;
            }
        }
    }

    ~parameter_tracker_t()
    {
        for (const auto& [param_name, count] : m_params_usage)
        {
            if (count == 0)
            {
                log_warning() << "parameter \"" << param_name << "\" was not used.";
            }
        }
    }

private:
    // attributes
    const cmdline_t::result_t& m_options;      ///<
    std::map<string_t, int>    m_params_usage; ///<
};
