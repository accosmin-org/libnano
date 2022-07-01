#include <nano/core/estimator.h>
#include <nano/core/logger.h>
#include <nano/core/stream.h>

using namespace nano;

template <typename tname>
static parameter_t* find_param(parameters_t& parameters, const tname& name, bool mandatory)
{
    const auto it = std::find_if(parameters.begin(), parameters.end(),
                                 [&](const parameter_t& param) { return param.name() == name; });

    critical(mandatory && (it == parameters.end()), "estimator: cannot find mandatory parameter (", name, ")!");

    return (it == parameters.end()) ? nullptr : (&*it);
}

template <typename tname>
static const parameter_t* find_param(const parameters_t& parameters, const tname& name, bool mandatory)
{
    const auto it = std::find_if(parameters.begin(), parameters.end(),
                                 [&](const parameter_t& param) { return param.name() == name; });

    critical(mandatory && (it == parameters.end()), "estimator: cannot find mandatory parameter (", name, ")!");

    return (it == parameters.end()) ? nullptr : (&*it);
}

void estimator_t::register_parameter(parameter_t parameter)
{
    critical(parameter_if(parameter.name()), "estimator: cannot register duplicated parameter (", parameter.name(),
             ")!");

    m_parameters.emplace_back(std::move(parameter));
}

parameter_t& estimator_t::parameter(const char* name)
{
    return *find_param(m_parameters, name, true);
}

parameter_t& estimator_t::parameter(const string_t& name)
{
    return *find_param(m_parameters, name, true);
}

const parameter_t& estimator_t::parameter(const char* name) const
{
    return *find_param(m_parameters, name, true);
}

const parameter_t& estimator_t::parameter(const string_t& name) const
{
    return *find_param(m_parameters, name, true);
}

parameter_t* estimator_t::parameter_if(const char* name)
{
    return find_param(m_parameters, name, false);
}

parameter_t* estimator_t::parameter_if(const string_t& name)
{
    return find_param(m_parameters, name, false);
}

const parameter_t* estimator_t::parameter_if(const char* name) const
{
    return find_param(m_parameters, name, false);
}

const parameter_t* estimator_t::parameter_if(const string_t& name) const
{
    return find_param(m_parameters, name, false);
}

std::istream& estimator_t::read(std::istream& stream)
{
    critical(!::nano::read(stream, m_major_version) || !::nano::read(stream, m_minor_version) ||
                 !::nano::read(stream, m_patch_version),
             "estimator: failed to read from stream!");

    critical(m_major_version > nano::major_version ||
                 (m_major_version == nano::major_version && m_minor_version > nano::minor_version) ||
                 (m_major_version == nano::major_version && m_minor_version == nano::minor_version &&
                  m_patch_version > nano::patch_version),
             "estimator: version mismatch!");

    critical(!::nano::read(stream, m_parameters), "estimator: failed to read from stream!");

    return stream;
}

std::ostream& estimator_t::write(std::ostream& stream) const
{
    critical(!::nano::write(stream, static_cast<int32_t>(nano::major_version)) ||
                 !::nano::write(stream, static_cast<int32_t>(nano::minor_version)) ||
                 !::nano::write(stream, static_cast<int32_t>(nano::patch_version)),
             "estimator: failed to write to stream");

    critical(!::nano::write(stream, m_parameters), "estimator: failed to write to stream!");

    return stream;
}

std::istream& nano::read(std::istream& stream, estimator_t& estimator)
{
    return estimator.read(stream);
}

std::ostream& nano::write(std::ostream& stream, const estimator_t& estimator)
{
    return estimator.write(stream);
}
