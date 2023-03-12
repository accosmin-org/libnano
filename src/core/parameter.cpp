#include <cmath>
#include <nano/core/logger.h>
#include <nano/core/numeric.h>
#include <nano/core/parameter.h>
#include <nano/core/stream.h>
#include <nano/core/tokenizer.h>

using namespace nano;

static auto name(const LEorLT& lelt)
{
    return std::holds_alternative<LE_t>(lelt) ? "<=" : "<";
}

template <typename tscalar>
static auto check(const LEorLT& lelt, tscalar value1, tscalar value2)
{
    return std::holds_alternative<LE_t>(lelt) ? (value1 <= value2) : (value1 < value2);
}

static auto split_pair(const string_t& value)
{
    string_t value1, value2;
    for (auto tokenizer = tokenizer_t{value, ";,:|/ "}; tokenizer; ++tokenizer)
    {
        (value1.empty() ? value1 : value2) = tokenizer.get();
    }
    return std::make_tuple(value1, value2);
}

static auto& update(const string_t& name, parameter_t::enum_t& param, string_t&& value)
{
    critical(std::find(param.m_domain.begin(), param.m_domain.end(), value) == param.m_domain.end(), "parameter (",
             name, "): out of domain enumeration value, !('", value, "' in [", scat(param.m_domain), "])");

    param.m_value = value;
    return param;
}

template <typename tscalar, typename tvalue>
static auto& update(const string_t& name, parameter_t::range_t<tscalar>& param, tvalue value_)
{
    const auto value = static_cast<tscalar>(value_);

    critical(!::nano::isfinite(value) || !::check(param.m_mincomp, param.m_min, value) ||
                 !::check(param.m_maxcomp, value, param.m_max),
             "parameter (", name, "): out of domain scalar value, !(", param.m_min, ::name(param.m_mincomp), value_,
             ::name(param.m_maxcomp), param.m_max, ")");

    param.m_value = value;
    return param;
}

template <typename tscalar, typename tvalue1, typename tvalue2>
static auto& update(const string_t& name, parameter_t::pair_range_t<tscalar>& param, tvalue1 value1_, tvalue2 value2_)
{
    const auto value1 = static_cast<tscalar>(value1_);
    const auto value2 = static_cast<tscalar>(value2_);

    critical(!::nano::isfinite(value1) || !::nano::isfinite(value2) || !::check(param.m_mincomp, param.m_min, value1) ||
                 !::check(param.m_valcomp, value1, value2) || !::check(param.m_maxcomp, value2, param.m_max),
             "parameter (", name, "): out of domain pair of scalar values, !(", param.m_min, ::name(param.m_mincomp),
             value1_, ::name(param.m_valcomp), value2_, ::name(param.m_maxcomp), param.m_max, ")");

    param.m_value1 = value1;
    param.m_value2 = value2;
    return param;
}

template <typename tvalue, std::enable_if_t<std::is_arithmetic_v<tvalue>, bool> = true>
static void update(const string_t& name, parameter_t::storage_t& storage, tvalue value)
{
    std::visit(overloaded{[&](parameter_t::irange_t& param) { ::update(name, param, value); },
                          [&](parameter_t::frange_t& param) { ::update(name, param, value); },
                          [&](auto&) { critical0("parameter (", name, "): cannot set value (", value, ")!"); }},
               storage);
}

template <typename tvalue, std::enable_if_t<std::is_arithmetic_v<tvalue>, bool> = true>
static void update(const string_t& name, parameter_t::storage_t& storage, std::tuple<tvalue, tvalue> value)
{
    const auto value1 = std::get<0>(value);
    const auto value2 = std::get<1>(value);

    std::visit(overloaded{[&](parameter_t::iprange_t& param) { ::update(name, param, value1, value2); },
                          [&](parameter_t::fprange_t& param) { ::update(name, param, value1, value2); },
                          [&](auto&)
                          { critical0("parameter (", name, "): cannot set value (", value1, ",", value2, ")!"); }},
               storage);
}

static auto make_comp(uint32_t flag)
{
    return (flag != 0U) ? LEorLT{LE} : LEorLT{LT};
}

static auto make_flag(LEorLT comp)
{
    return std::get_if<LE_t>(&comp) != nullptr ? 1U : 0U;
}

template <typename tscalar>
static auto read(const string_t& name, std::istream& stream, parameter_t::range_t<tscalar>)
{
    tscalar  value = 0, min = 0, max = 0;
    uint32_t minLE = 0U, maxLE = 0U;

    critical(!::nano::read(stream, value) ||     // LCOV_EXCL_LINE
                 !::nano::read(stream, min) ||   // LCOV_EXCL_LINE
                 !::nano::read(stream, max) ||   // LCOV_EXCL_LINE
                 !::nano::read(stream, minLE) || // LCOV_EXCL_LINE
                 !::nano::read(stream, maxLE),   // LCOV_EXCL_LINE
             "parameter (", name, "): failed to read from stream!");

    return parameter_t::range_t<tscalar>{value, min, max, make_comp(minLE), make_comp(maxLE)};
}

template <typename tscalar>
static auto read(const string_t& name, std::istream& stream, parameter_t::pair_range_t<tscalar>)
{
    tscalar  value1 = 0, value2 = 0, min = 0, max = 0;
    uint32_t minLE = 0U, maxLE = 0U, valueLE = 0U;

    critical(!::nano::read(stream, value1) ||     // LCOV_EXCL_LINE
                 !::nano::read(stream, value2) || // LCOV_EXCL_LINE
                 !::nano::read(stream, min) ||    // LCOV_EXCL_LINE
                 !::nano::read(stream, max) ||    // LCOV_EXCL_LINE
                 !::nano::read(stream, minLE) ||  // LCOV_EXCL_LINE
                 !::nano::read(stream, maxLE) ||  // LCOV_EXCL_LINE
                 !::nano::read(stream, valueLE),  // LCOV_EXCL_LINE
             "parameter (", name, "): failed to read from stream!");

    return parameter_t::pair_range_t<tscalar>{value1,          value2, min, max, make_comp(minLE), make_comp(valueLE),
                                              make_comp(maxLE)};
}

template <typename tscalar>
static void write(const string_t& name, std::ostream& stream, int32_t type, const parameter_t::range_t<tscalar>& param)
{
    critical(!::nano::write(stream, type) || !::nano::write(stream, name) ||
                 !::nano::write(stream, param.m_value) ||              // LCOV_EXCL_LINE
                 !::nano::write(stream, param.m_min) ||                // LCOV_EXCL_LINE
                 !::nano::write(stream, param.m_max) ||                // LCOV_EXCL_LINE
                 !::nano::write(stream, make_flag(param.m_mincomp)) || // LCOV_EXCL_LINE
                 !::nano::write(stream, make_flag(param.m_maxcomp)),   // LCOV_EXCL_LINE
             "parameter (", name, "): failed to write to stream!");
}

template <typename tscalar>
static void write(const string_t& name, std::ostream& stream, int32_t type,
                  const parameter_t::pair_range_t<tscalar>& param)
{
    critical(!::nano::write(stream, type) || !::nano::write(stream, name) ||
                 !::nano::write(stream, param.m_value1) ||             // LCOV_EXCL_LINE
                 !::nano::write(stream, param.m_value2) ||             // LCOV_EXCL_LINE
                 !::nano::write(stream, param.m_min) ||                // LCOV_EXCL_LINE
                 !::nano::write(stream, param.m_max) ||                // LCOV_EXCL_LINE
                 !::nano::write(stream, make_flag(param.m_mincomp)) || // LCOV_EXCL_LINE
                 !::nano::write(stream, make_flag(param.m_maxcomp)) || // LCOV_EXCL_LINE
                 !::nano::write(stream, make_flag(param.m_valcomp)),   // LCOV_EXCL_LINE
             "parameter (", name, "): failed to write to stream!");
}

static bool operator==(const parameter_t::enum_t& lhs, const parameter_t::enum_t& rhs)
{
    return lhs.m_value == rhs.m_value && lhs.m_domain == rhs.m_domain;
}

template <typename tscalar>
static bool operator==(const parameter_t::range_t<tscalar>& lhs, const parameter_t::range_t<tscalar>& rhs)
{
    return lhs.m_value == rhs.m_value && lhs.m_min == rhs.m_min &&
           make_flag(lhs.m_mincomp) == make_flag(rhs.m_mincomp) && lhs.m_max == rhs.m_max &&
           make_flag(lhs.m_maxcomp) == make_flag(rhs.m_maxcomp);
}

template <typename tscalar>
static bool operator==(const parameter_t::pair_range_t<tscalar>& lhs, const parameter_t::pair_range_t<tscalar>& rhs)
{
    return lhs.m_value1 == rhs.m_value1 && lhs.m_value2 == rhs.m_value2 &&
           make_flag(lhs.m_valcomp) == make_flag(rhs.m_valcomp) && lhs.m_min == rhs.m_min &&
           make_flag(lhs.m_mincomp) == make_flag(rhs.m_mincomp) && lhs.m_max == rhs.m_max &&
           make_flag(lhs.m_maxcomp) == make_flag(rhs.m_maxcomp);
}

template <typename tparam>
static bool operator==(const tparam& lparam, const parameter_t::storage_t& rstorage)
{
    const auto* rparam = std::get_if<tparam>(&rstorage);
    return rparam != nullptr && lparam == *rparam;
}

static std::ostream& value(std::ostream& stream, const parameter_t::enum_t& param)
{
    return stream << param.m_value;
}

static std::ostream& domain(std::ostream& stream, const parameter_t::enum_t& param)
{
    return stream << scat(param.m_domain);
}

template <typename tscalar>
static std::ostream& value(std::ostream& stream, const parameter_t::range_t<tscalar>& param)
{
    return stream << param.m_value;
}

template <typename tscalar>
static std::ostream& domain(std::ostream& stream, const parameter_t::range_t<tscalar>& param)
{
    return stream << param.m_min << " " << ::name(param.m_mincomp) << " " << param.m_value << " "
                  << ::name(param.m_maxcomp) << " " << param.m_max;
}

template <typename tscalar>
static std::ostream& value(std::ostream& stream, const parameter_t::pair_range_t<tscalar>& param)
{
    return stream << "(" << param.m_value1 << "," << param.m_value2 << ")";
}

template <typename tscalar>
static std::ostream& domain(std::ostream& stream, const parameter_t::pair_range_t<tscalar>& param)
{
    return stream << param.m_min << " " << ::name(param.m_mincomp) << " " << param.m_value1 << " "
                  << ::name(param.m_valcomp) << " " << param.m_value2 << " " << ::name(param.m_maxcomp) << " "
                  << param.m_max;
}

parameter_t::parameter_t() = default;

parameter_t::parameter_t(string_t name, enum_t param)
    : m_name(std::move(name))
    , m_storage(std::move(::update(m_name, param, std::move(param.m_value))))
{
}

parameter_t::parameter_t(string_t name, irange_t param)
    : m_name(std::move(name))
    , m_storage(::update(m_name, param, param.m_value))
{
}

parameter_t::parameter_t(string_t name, frange_t param)
    : m_name(std::move(name))
    , m_storage(::update(m_name, param, param.m_value))
{
}

parameter_t::parameter_t(string_t name, iprange_t param)
    : m_name(std::move(name))
    , m_storage(::update(m_name, param, param.m_value1, param.m_value2))
{
}

parameter_t::parameter_t(string_t name, fprange_t param)
    : m_name(std::move(name))
    , m_storage(::update(m_name, param, param.m_value1, param.m_value2))
{
}

parameter_t& parameter_t::seti(int64_t value)
{
    ::update(m_name, m_storage, value);
    return *this;
}

parameter_t& parameter_t::setd(scalar_t value)
{
    ::update(m_name, m_storage, value);
    return *this;
}

parameter_t& parameter_t::operator=(string_t value)
{
    std::visit(overloaded{[&](enum_t& param) { ::update(m_name, param, std::move(value)); },
                          [&](irange_t& param) { ::update(m_name, param, std::stoll(value)); },
                          [&](frange_t& param) { ::update(m_name, param, std::stod(value)); },
                          [&](iprange_t& param)
                          {
                              const auto [value1, value2] = ::split_pair(value);
                              ::update(m_name, param, std::stoll(value1), std::stoll(value2));
                          },
                          [&](fprange_t& param)
                          {
                              const auto [value1, value2] = ::split_pair(value);
                              ::update(m_name, param, std::stod(value1), std::stod(value2));
                          },
                          [&](auto&) { critical0("parameter (", m_name, "): cannot set value (", value, ")!"); }},
               m_storage);
    return *this;
}

parameter_t& parameter_t::operator=(std::tuple<int32_t, int32_t> value)
{
    ::update(m_name, m_storage, value);
    return *this;
}

parameter_t& parameter_t::operator=(std::tuple<int64_t, int64_t> value)
{
    ::update(m_name, m_storage, value);
    return *this;
}

parameter_t& parameter_t::operator=(std::tuple<scalar_t, scalar_t> value)
{
    ::update(m_name, m_storage, value);
    return *this;
}

std::istream& parameter_t::read(std::istream& stream)
{
    int32_t type = -1;
    critical(!::nano::read(stream, type) || !::nano::read(stream, m_name), "parameter (", m_name,
             "): failed to read from stream!");

    switch (type)
    {
    case -1:
    {
        m_storage = storage_t{};
    }
    break;
    case 0:
    {
        string_t  value;
        strings_t domain;

        critical(!::nano::read(stream, value) ||    // LCOV_EXCL_LINE
                     !::nano::read(stream, domain), // LCOV_EXCL_LINE
                 "parameter (", m_name, "): failed to read from stream!");
        m_storage = enum_t{value, domain};
    }
    break;
    case 1: m_storage = ::read(m_name, stream, irange_t{}); break;
    case 2: m_storage = ::read(m_name, stream, frange_t{}); break;
    case 3: m_storage = ::read(m_name, stream, iprange_t{}); break;
    case 4: m_storage = ::read(m_name, stream, fprange_t{}); break;
    default: critical0("parameter (", m_name, "): failed to read from stream (type=", type, ")!");
    }

    return stream;
}

std::ostream& parameter_t::write(std::ostream& stream) const
{
    std::visit(overloaded{[&](const std::monostate&)
                          {
                              const int32_t type = -1;
                              critical(!::nano::write(stream, type) || !::nano::write(stream, m_name), "parameter (",
                                       m_name, "): failed to write to stream!");
                          },
                          [&](const enum_t& param)
                          {
                              const int32_t type = 0;
                              critical(!::nano::write(stream, type) || !::nano::write(stream, m_name) ||
                                           !::nano::write(stream, param.m_value) || // LCOV_EXCL_LINE
                                           !::nano::write(stream, param.m_domain),  // LCOV_EXCL_LINE
                                       "parameter (", m_name, "): failed to write to stream!");
                          },
                          [&](const irange_t& param) { ::write(m_name, stream, 1, param); },
                          [&](const frange_t& param) { ::write(m_name, stream, 2, param); },
                          [&](const iprange_t& param) { ::write(m_name, stream, 3, param); },
                          [&](const fprange_t& param) { ::write(m_name, stream, 4, param); }},
               m_storage);

    return stream;
}

void parameter_t::logical_error() const
{
    critical0("parameter (", m_name, "): logical error, unexpected parameter type!");
}

bool nano::operator==(const parameter_t& lhs, const parameter_t& rhs)
{
    using ::operator==;
    return lhs.name() == rhs.name() &&
           std::visit(overloaded{[&rhs](const std::monostate&)
                                 { return std::get_if<std::monostate>(&rhs.storage()) != nullptr; },
                                 [&rhs](const parameter_t::enum_t& lparam) { return lparam == rhs.storage(); },
                                 [&rhs](const parameter_t::irange_t& lparam) { return lparam == rhs.storage(); },
                                 [&rhs](const parameter_t::frange_t& lparam) { return lparam == rhs.storage(); },
                                 [&rhs](const parameter_t::iprange_t& lparam) { return lparam == rhs.storage(); },
                                 [&rhs](const parameter_t::fprange_t& lparam) { return lparam == rhs.storage(); }},
                      lhs.storage());
}

bool nano::operator!=(const parameter_t& lhs, const parameter_t& rhs)
{
    return !(lhs == rhs);
}

std::ostream& nano::operator<<(std::ostream& stream, const parameter_t& parameter)
{
    return stream << parameter.name() << "=" << parameter.value() << "|domain=[" << parameter.domain() << "]";
}

std::ostream& nano::operator<<(std::ostream& stream, const parameter_t::value_t& value)
{
    std::visit(overloaded{[&stream](const std::monostate&) { stream << "N/A"; },
                          [&stream](const parameter_t::enum_t& param) { ::value(stream, param); },
                          [&stream](const parameter_t::irange_t& param) { ::value(stream, param); },
                          [&stream](const parameter_t::frange_t& param) { ::value(stream, param); },
                          [&stream](const parameter_t::iprange_t& param) { ::value(stream, param); },
                          [&stream](const parameter_t::fprange_t& param) { ::value(stream, param); }},
               value.m_parameter.storage());
    return stream;
}

std::ostream& nano::operator<<(std::ostream& stream, const parameter_t::domain_t& domain)
{
    std::visit(overloaded{[&stream](const std::monostate&) { stream << "N/A"; },
                          [&stream](const parameter_t::enum_t& param) { ::domain(stream, param); },
                          [&stream](const parameter_t::irange_t& param) { ::domain(stream, param); },
                          [&stream](const parameter_t::frange_t& param) { ::domain(stream, param); },
                          [&stream](const parameter_t::iprange_t& param) { ::domain(stream, param); },
                          [&stream](const parameter_t::fprange_t& param) { ::domain(stream, param); }},
               domain.m_parameter.storage());
    return stream;
}
