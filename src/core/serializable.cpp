#include <nano/core/stream.h>
#include <nano/core/serializable.h>

using namespace nano;

template <typename tname>
static parameter_t& find_param(parameters_t& parameters, const tname& name)
{
    const auto it = std::find_if(parameters.begin(), parameters.end(), [&] (const parameter_t& param)
    {
        return param.name() == name;
    });

    critical(it == parameters.end(), "serializable: cannot find parameter by name (", name, ")!");
    return *it;
}

template <typename tname>
static const parameter_t& find_param(const parameters_t& parameters, const tname& name)
{
    const auto it = std::find_if(parameters.begin(), parameters.end(), [&] (const parameter_t& param)
    {
        return param.name() == name;
    });

    critical(it == parameters.end(), "serializable: cannot find parameter by name (", name, ")!");
    return *it;
}

void serializable_t::register_param(eparam1_t param)
{
    m_parameters.emplace_back(std::move(param));
}

void serializable_t::register_param(iparam1_t param)
{
    m_parameters.emplace_back(std::move(param));
}

void serializable_t::register_param(sparam1_t param)
{
    m_parameters.emplace_back(std::move(param));
}

void serializable_t::set(const char* name, int32_t value)
{
    find(name).set(value);
}

void serializable_t::set(const char* name, int64_t value)
{
    find(name).set(value);
} // LCOV_EXCL_LINE

void serializable_t::set(const char* name, scalar_t value)
{
    find(name).set(value);
}

void serializable_t::set(const string_t& name, int32_t value)
{
    find(name).set(value);
}

void serializable_t::set(const string_t& name, int64_t value)
{
    find(name).set(value);
} // LCOV_EXCL_LINE

void serializable_t::set(const string_t& name, scalar_t value)
{
    find(name).set(value);
}

int64_t serializable_t::ivalue(const char* name) const
{
    return find(name).ivalue();
}

int64_t serializable_t::ivalue(const string_t& name) const
{
    return find(name).ivalue();
}

scalar_t serializable_t::svalue(const char* name) const
{
    return find(name).svalue();
}

scalar_t serializable_t::svalue(const string_t& name) const
{
    return find(name).svalue();
}

parameter_t& serializable_t::find(const char* name)
{
    return find_param(m_parameters, name);
}

parameter_t& serializable_t::find(const string_t& name)
{
    return find_param(m_parameters, name);
}

const parameter_t& serializable_t::find(const char* name) const
{
    return find_param(m_parameters, name);
}

const parameter_t& serializable_t::find(const string_t& name) const
{
    return find_param(m_parameters, name);
}

void serializable_t::read(std::istream& stream)
{
    critical(
        !::nano::read(stream, m_major_version) ||
        !::nano::read(stream, m_minor_version) ||
        !::nano::read(stream, m_patch_version),
        "serializable: failed to read from stream!");

    critical(
        m_major_version > nano::major_version ||
        (m_major_version == nano::major_version &&
         m_minor_version > nano::minor_version) ||
        (m_major_version == nano::major_version &&
         m_minor_version == nano::minor_version &&
         m_patch_version > nano::patch_version),
        "serializable: version mismatch!");

    critical(
        !::nano::read(stream, m_parameters),
        "serializable: failed to read from stream!");
}

void serializable_t::write(std::ostream& stream) const
{
    critical(
        !::nano::write(stream, static_cast<int32_t>(nano::major_version)) ||
        !::nano::write(stream, static_cast<int32_t>(nano::minor_version)) ||
        !::nano::write(stream, static_cast<int32_t>(nano::patch_version)),
        "serializable: failed to write to stream");

    critical(
        !::nano::write(stream, m_parameters),
        "serializable: failed to write to stream!");
}

std::istream& nano::read(std::istream& stream, serializable_t& object)
{
    object.read(stream);
    return stream;
}

std::ostream& nano::write(std::ostream& stream, const serializable_t& object)
{
    object.write(stream);
    return stream;
}
