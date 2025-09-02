#pragma once

#include <nano/arch.h>
#include <nano/core/overloaded.h>
#include <nano/core/strutil.h>
#include <nano/scalar.h>
#include <variant>

namespace nano
{
///
/// \brief less or equal operator.
///
struct LE_t
{
};

///
/// \brief less than operator.
///
struct LT_t
{
};

static const auto LE = LE_t{};
static const auto LT = LT_t{};

using LEorLT = std::variant<LE_t, LT_t>;

///
/// \brief named parameter with support for:
///     - automatic validity check (e.g. in a given range)
///     - serialization to and from binary streams.
///
class NANO_PUBLIC parameter_t
{
public:
    struct value_t
    {
        const parameter_t& m_parameter;
    };

    struct domain_t
    {
        const parameter_t& m_parameter;
    };

    struct enum_t
    {
        string_t  m_value;  ///< the stored value as string
        strings_t m_domain; ///< domain of available values as strings
    };

    template <class tscalar>
    requires std::is_arithmetic_v<tscalar>
    struct range_t
    {
        template <class tvalue>
        auto value() const
        {
            return static_cast<tvalue>(m_value);
        }

        tscalar m_value{0}; ///< the stored value
        tscalar m_min{0};   ///< the allowed [min, max] range
        tscalar m_max{0};   ///< the allowed [min, max] range
        LEorLT  m_mincomp;  ///<
        LEorLT  m_maxcomp;  ///<
    };

    template <class tscalar>
    requires std::is_arithmetic_v<tscalar>
    struct pair_range_t
    {
        template <class tvalue>
        auto value() const
        {
            return std::make_tuple(static_cast<tvalue>(m_value1), static_cast<tvalue>(m_value2));
        }

        tscalar m_value1{0}; ///< the stored values
        tscalar m_value2{0}; ///< the stored values
        tscalar m_min{0};    ///< the allowed [min, max] range
        tscalar m_max{0};    ///< the allowed [min, max] range
        LEorLT  m_mincomp;   ///<
        LEorLT  m_valcomp;   ///<
        LEorLT  m_maxcomp;   ///<
    };

    using irange_t  = range_t<int64_t>;
    using frange_t  = range_t<scalar_t>;
    using iprange_t = pair_range_t<int64_t>;
    using fprange_t = pair_range_t<scalar_t>;

    using storage_t = std::variant<std::monostate, enum_t, irange_t, frange_t, iprange_t, fprange_t, string_t>;

    ///
    /// \brief default constructor
    ///
    parameter_t();

    ///
    /// \brief return a constrained enumeration parameter.
    ///
    template <class tenum>
    requires std::is_enum_v<tenum>
    static parameter_t make_enum(string_t name, tenum value)
    {
        return make_enum_(std::move(name), value);
    }

    ///
    /// \brief return a string parameter.
    ///
    static parameter_t make_string(string_t name, string_t value) { return {std::move(name), std::move(value)}; }

    ///
    /// \brief return a floating point parameter constrained to the given range:
    ///     min LE/LT value LE/LT max.
    ///
    template <class tmin, class tvalue, class tmax>
    requires(std::is_arithmetic_v<tmin> && std::is_arithmetic_v<tvalue> && std::is_arithmetic_v<tmax>)
    static parameter_t make_scalar(string_t name, tmin min, LEorLT mincomp, tvalue value, LEorLT maxcomp, tmax max)
    {
        return make_scalar_<scalar_t>(std::move(name), min, mincomp, value, maxcomp, max);
    }

    ///
    /// \brief return an integer parameter constrained to the given range:
    ///     min LE/LT value LE/LT max.
    ///
    template <class tmin, class tvalue, class tmax>
    requires(std::is_arithmetic_v<tmin> && std::is_arithmetic_v<tvalue> && std::is_arithmetic_v<tmax>)
    static parameter_t make_integer(string_t name, tmin min, LEorLT mincomp, tvalue value, LEorLT maxcomp, tmax max)
    {
        return make_scalar_<int64_t>(std::move(name), min, mincomp, value, maxcomp, max);
    }

    ///
    /// \brief return an ordered floating point pair parameter constrained to the given range:
    ///     min LE/LT value1 LE/LT value2 LE/LT max.
    ///
    template <class tmin, class tvalue1, class tvalue2, class tmax>
    requires(std::is_arithmetic_v<tmin> && std::is_arithmetic_v<tvalue1> && std::is_arithmetic_v<tvalue2> &&
             std::is_arithmetic_v<tmax>)
    static parameter_t make_scalar_pair(string_t name, tmin min, LEorLT mincomp, tvalue1 value1, LEorLT valcomp,
                                        tvalue2 value2, LEorLT maxcomp, tmax max)
    {
        return make_scalar_<scalar_t>(std::move(name), min, mincomp, value1, valcomp, value2, maxcomp, max);
    }

    ///
    /// \brief return an integer pair parameter constrained to the given range:
    ///     min LE/LT value1 LE/LT value2 LE/LT max.
    ///
    template <class tmin, class tvalue1, class tvalue2, class tmax>
    requires(std::is_arithmetic_v<tmin> && std::is_arithmetic_v<tvalue1> && std::is_arithmetic_v<tvalue2> &&
             std::is_arithmetic_v<tmax>)
    static parameter_t make_integer_pair(string_t name, tmin min, LEorLT mincomp, tvalue1 value1, LEorLT valcomp,
                                         tvalue2 value2, LEorLT maxcomp, tmax max)
    {
        return make_scalar_<int64_t>(std::move(name), min, mincomp, value1, valcomp, value2, maxcomp, max);
    }

    ///
    /// \brief change the parameter's value, throws an exception if not possible.
    ///
    parameter_t& operator=(string_t);
    parameter_t& operator=(std::tuple<int32_t, int32_t>);
    parameter_t& operator=(std::tuple<int64_t, int64_t>);
    parameter_t& operator=(std::tuple<scalar_t, scalar_t>);

    template <class tscalar>
    requires std::is_arithmetic_v<tscalar>
    parameter_t& operator=(const tscalar value)
    {
        if constexpr (std::is_integral_v<tscalar>)
        {
            seti(static_cast<int64_t>(value));
        }
        else
        {
            setd(static_cast<scalar_t>(value));
        }
        return *this;
    }

    template <class tenum>
    requires std::is_enum_v<tenum>
    parameter_t& operator=(const tenum value)
    {
        if (std::get_if<enum_t>(&m_storage))
        {
            this->operator=(scat(value));
        }
        else
        {
            logical_error();
        }
        return *this;
    }

    ///
    /// \brief retrieve the current parameter's value.
    ///
    template <class tstring>
    requires std::is_same_v<tstring, string_t>
    string_t value() const
    {
        if (const auto* param = std::get_if<string_t>(&m_storage))
        {
            return *param;
        }
        logical_error();
    }

    template <class tenum>
    requires std::is_enum_v<tenum>
    tenum value() const
    {
        if (const auto* param = std::get_if<enum_t>(&m_storage))
        {
            return from_string<tenum>(param->m_value);
        }
        logical_error();
    }

    template <class tscalar>
    requires std::is_arithmetic_v<tscalar>
    tscalar value() const
    {
        return std::visit(overloaded{[](const irange_t& param) { return param.value<tscalar>(); },
                                     [](const frange_t& param) { return param.value<tscalar>(); },
                                     [this](const auto&)
                                     {
                                         logical_error();
                                         return tscalar{};
                                     }},
                          m_storage);
    }

    template <class tscalar>
    requires std::is_arithmetic_v<tscalar>
    std::tuple<tscalar, tscalar> value_pair() const
    {
        return std::visit(overloaded{[](const iprange_t& param) { return param.value<tscalar>(); },
                                     [](const fprange_t& param) { return param.value<tscalar>(); },
                                     [this](const auto&)
                                     {
                                         logical_error();
                                         return std::tuple<tscalar, tscalar>{};
                                     }},
                          m_storage);
    }

    ///
    /// \brief serialize from the given binary stream.
    ///
    /// NB: any error is considered critical and expected to result in an exception.
    ///
    std::istream& read(std::istream&);

    ///
    /// \brief serialize to the given binary stream.
    ///
    /// NB: any error is considered critical and expected to result in an exception.
    ///
    std::ostream& write(std::ostream&) const;

    ///
    /// \brief return the parameter's name.
    ///
    const string_t& name() const { return m_name; }

    ///
    /// \brief return the storage container of the parameter's value.
    ///
    const storage_t& storage() const { return m_storage; }

    ///
    /// \brief return a typed object representing the parameter's value.
    ///
    value_t value() const { return value_t{*this}; }

    ///
    /// \brief return a typed object representing the parameter's domain.
    ///
    domain_t domain() const { return domain_t{*this}; }

private:
    parameter_t& seti(int64_t);
    parameter_t& setd(scalar_t);
    parameter_t(string_t name, enum_t);
    parameter_t(string_t name, string_t);
    parameter_t(string_t name, irange_t);
    parameter_t(string_t name, frange_t);
    parameter_t(string_t name, iprange_t);
    parameter_t(string_t name, fprange_t);

    [[noreturn]] void logical_error() const;

    template <class tenum>
    static parameter_t make_enum_(string_t name, tenum value)
    {
        static const auto options = enum_string<tenum>();

        strings_t domain{options.size()};
        std::transform(options.begin(), options.end(), domain.begin(), [](const auto& v) { return v.second; });

        return parameter_t{
            std::move(name), enum_t{scat(value), std::move(domain)}
        };
    }

    template <class tscalar, class tmin, class tvalue, class tmax>
    static parameter_t make_scalar_(const string_t& name, tmin min, LEorLT mincomp, tvalue value, LEorLT maxcomp,
                                    tmax max)
    {
        return parameter_t{
            name, range_t<tscalar>{static_cast<tscalar>(value), static_cast<tscalar>(min), static_cast<tscalar>(max),
                                   mincomp, maxcomp}
        };
    }

    template <class tscalar, class tmin, class tvalue1, class tvalue2, class tmax>
    static parameter_t make_scalar_(const string_t& name, tmin min, LEorLT mincomp, tvalue1 value1, LEorLT valcomp,
                                    tvalue2 value2, LEorLT maxcomp, tmax max)
    {
        return parameter_t{
            name,
            pair_range_t<tscalar>{static_cast<tscalar>(value1), static_cast<tscalar>(value2), static_cast<tscalar>(min),
                                  static_cast<tscalar>(max), mincomp, valcomp, maxcomp}
        };
    }

    // attributes
    string_t  m_name;    ///<
    storage_t m_storage; ///<
};

using parameters_t = std::vector<parameter_t>;

NANO_PUBLIC bool operator==(const parameter_t&, const parameter_t&);
NANO_PUBLIC bool operator!=(const parameter_t&, const parameter_t&);
NANO_PUBLIC std::ostream& operator<<(std::ostream&, const parameter_t&);
NANO_PUBLIC std::ostream& operator<<(std::ostream&, const parameter_t::value_t&);
NANO_PUBLIC std::ostream& operator<<(std::ostream&, const parameter_t::domain_t&);
} // namespace nano
