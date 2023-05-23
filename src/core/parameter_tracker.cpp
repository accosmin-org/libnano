#include <nano/core/logger.h>
#include <nano/core/parameter_tracker.h>

using namespace nano;

parameter_tracker_t::parameter_tracker_t(const cmdline_t::result_t& options)
    : m_options(options)
{
    for (const auto& [param_name, param_value] : m_options.m_xvalues)
    {
        m_params_usage[param_name] = 0;
    }
}

void parameter_tracker_t::setup(configurable_t& configurable)
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

parameter_tracker_t::~parameter_tracker_t()
{
    for (const auto& [param_name, count] : m_params_usage)
    {
        if (count == 0)
        {
            log_warning() << "parameter \"" << param_name << "\" was not used.";
        }
    }
}
