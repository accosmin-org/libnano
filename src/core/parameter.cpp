#include <nano/core/stream.h>
#include <nano/core/parameter.h>

using namespace nano;

eparam1_t::eparam1_t() = default;

eparam1_t::eparam1_t(string_t name, string_t value, strings_t domain) :
    m_name(std::move(name)),
    m_domain(std::move(domain))
{
    set(std::move(value));
}

void eparam1_t::set(string_t value)
{
    critical(
        std::find(m_domain.begin(), m_domain.end(), value) == m_domain.end(),
        "invalid parameter '", m_name, "': (", value, ") not in domain (", scat(m_domain), ")!");

    m_value = std::move(value);
}

parameter_t::parameter_t(eparam1_t param) :
    m_storage(std::move(param))
{
}

parameter_t::parameter_t(iparam1_t param) :
    m_storage(std::move(param))
{
}

parameter_t::parameter_t(sparam1_t param) :
    m_storage(std::move(param))
{
}

void parameter_t::set(int32_t value)
{
    set(static_cast<int64_t>(value));
}

void parameter_t::set(int64_t value)
{
    if (is_ivalue())
    {
        iparam().set(value);
    }
    else if (is_svalue())
    {
        sparam().set(static_cast<scalar_t>(value));
    }
    else
    {
        critical0("parameter (", name(), "): cannot set enumeration with integer (", value, ")!");
    }
}

void parameter_t::set(scalar_t value)
{
    if (is_svalue())
    {
        sparam().set(value);
    }
    else
    {
        critical0("parameter (", name(), "): cannot set not-scalar with scalar (", value, ")!");
    }
}

int64_t parameter_t::ivalue() const
{
    return iparam().get();
}

scalar_t parameter_t::svalue() const
{
    return sparam().get();
}

bool parameter_t::is_evalue() const
{
    return std::holds_alternative<eparam1_t>(m_storage);
}

bool parameter_t::is_ivalue() const
{
    return std::holds_alternative<iparam1_t>(m_storage);
}

bool parameter_t::is_svalue() const
{
    return std::holds_alternative<sparam1_t>(m_storage);
}

const string_t& parameter_t::name() const
{
    if (is_evalue())
    {
        return eparam().name();
    }
    else if (is_ivalue())
    {
        return iparam().name();
    }
    else
    {
        return sparam().name();
    }
}

void parameter_t::read(std::istream& stream)
{
    int32_t type = -1;
    critical(
        !::nano::read(stream, type),
        "parameter: failed to read from stream!");

    switch (type)
    {
    case 0:
        {
            string_t name, value;
            strings_t domain;

            critical(
                !::nano::read(stream, name) ||      // LCOV_EXCL_LINE
                !::nano::read(stream, value) ||     // LCOV_EXCL_LINE
                !::nano::read(stream, domain),      // LCOV_EXCL_LINE
                "parameter: failed to read from stream!");
            m_storage = eparam1_t{name, value, domain};
        }
        break;

    case 1:
        {
            string_t name;
            int64_t value = 0, min = 0, max = 0;
            uint32_t minLE = 0U, maxLE = 0U;

            critical(
                !::nano::read(stream, name) ||      // LCOV_EXCL_LINE
                !::nano::read(stream, value) ||     // LCOV_EXCL_LINE
                !::nano::read(stream, min) ||       // LCOV_EXCL_LINE
                !::nano::read(stream, max) ||       // LCOV_EXCL_LINE
                !::nano::read(stream, minLE) ||     // LCOV_EXCL_LINE
                !::nano::read(stream, maxLE),       // LCOV_EXCL_LINE
                "parameter: failed to read from stream!");

            m_storage = iparam1_t{name,
                min, (minLE != 0U) ? LEorLT{LE} : LEorLT{LT}, value, (maxLE != 0U) ? LEorLT{LE} : LEorLT{LT}, max};
        }
        break;

    case 2:
        {
            string_t name;
            scalar_t value = 0, min = 0, max = 0;
            uint32_t minLE = 0U, maxLE = 0U;

            critical(
                !::nano::read(stream, name) ||      // LCOV_EXCL_LINE
                !::nano::read(stream, value) ||     // LCOV_EXCL_LINE
                !::nano::read(stream, min) ||       // LCOV_EXCL_LINE
                !::nano::read(stream, max) ||       // LCOV_EXCL_LINE
                !::nano::read(stream, minLE) ||     // LCOV_EXCL_LINE
                !::nano::read(stream, maxLE),       // LCOV_EXCL_LINE
                "parameter: failed to read from stream!");

            m_storage = sparam1_t{name,
                min, (minLE != 0U) ? LEorLT{LE} : LEorLT{LT}, value, (maxLE != 0U) ? LEorLT{LE} : LEorLT{LT}, max};
        }
        break;

    default:
        critical0("parameter: failed to read from stream (type=", type, ")!");
        break;
    }
}

void parameter_t::write(std::ostream& stream) const
{
    int32_t type = 0;
    if (is_evalue())
    {
        const auto& param = eparam();

        type = 0;
        critical(
            !::nano::write(stream, type) ||
            !::nano::write(stream, param.name()) ||                             // LCOV_EXCL_LINE
            !::nano::write(stream, param.get()) ||                              // LCOV_EXCL_LINE
            !::nano::write(stream, param.domain()),                             // LCOV_EXCL_LINE
            "parameter (", name(), "): failed to write to stream!");
    }
    else if (is_ivalue())
    {
        const auto& param = iparam();

        type = 1;
        critical(
            !::nano::write(stream, type) ||
            !::nano::write(stream, param.name()) ||                             // LCOV_EXCL_LINE
            !::nano::write(stream, param.get()) ||                              // LCOV_EXCL_LINE
            !::nano::write(stream, param.min()) ||                              // LCOV_EXCL_LINE
            !::nano::write(stream, param.max()) ||                              // LCOV_EXCL_LINE
            !::nano::write(stream, static_cast<uint32_t>(param.minLE())) ||     // LCOV_EXCL_LINE
            !::nano::write(stream, static_cast<uint32_t>(param.maxLE())),       // LCOV_EXCL_LINE
            "parameter (", name(), "): failed to write to stream!");
    }
    else
    {
        const auto& param = sparam();

        type = 2;
        critical(
            !::nano::write(stream, type) ||
            !::nano::write(stream, param.name()) ||                             // LCOV_EXCL_LINE
            !::nano::write(stream, param.get()) ||                              // LCOV_EXCL_LINE
            !::nano::write(stream, param.min()) ||                              // LCOV_EXCL_LINE
            !::nano::write(stream, param.max()) ||                              // LCOV_EXCL_LINE
            !::nano::write(stream, static_cast<uint32_t>(param.minLE())) ||     // LCOV_EXCL_LINE
            !::nano::write(stream, static_cast<uint32_t>(param.maxLE())),       // LCOV_EXCL_LINE
            "parameter (", name(), "): failed to write to stream!");
    }
}

eparam1_t& parameter_t::eparam()
{
    critical(!is_evalue(), "parameter: expecting enumeration!");
    return std::get<eparam1_t>(m_storage);
}

const eparam1_t& parameter_t::eparam() const
{
    critical(!is_evalue(), "parameter: expecting enumeration!");
    return std::get<eparam1_t>(m_storage);
}

iparam1_t& parameter_t::iparam()
{
    critical(!is_ivalue(), "parameter: expecting integer!");
    return std::get<iparam1_t>(m_storage);
}

const iparam1_t& parameter_t::iparam() const
{
    critical(!is_ivalue(), "parameter: expecting integer!");
    return std::get<iparam1_t>(m_storage);
}

sparam1_t& parameter_t::sparam()
{
    critical(!is_svalue(), "parameter: expecting scalar!");
    return std::get<sparam1_t>(m_storage);
}

const sparam1_t& parameter_t::sparam() const
{
    critical(!is_svalue(), "parameter: expecting scalar!");
    return std::get<sparam1_t>(m_storage);
}

std::ostream& nano::operator<<(std::ostream& stream, const parameter_t& param)
{
    stream << param.name() << "=";
    if (param.is_svalue())
    {
        return stream << param.sparam().get();
    }
    else if (param.is_ivalue())
    {
        return stream << param.iparam().get();
    }
    else
    {
        return stream << param.eparam().get();
    }
}

std::istream& nano::read(std::istream& stream, parameter_t& object)
{
    object.read(stream);
    return stream;
}

std::ostream& nano::write(std::ostream& stream, const parameter_t& object)
{
    object.write(stream);
    return stream;
}
