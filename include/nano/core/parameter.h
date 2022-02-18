#pragma once

#include <variant>
#include <nano/arch.h>
#include <nano/scalar.h>
#include <nano/core/strutil.h>

namespace nano
{
    ///
    /// \brief less or equal operator.
    ///
    struct LE_t{};

    ///
    /// \brief less than operator.
    ///
    struct LT_t{};

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
            const parameter_t&  m_parameter;
        };
        struct domain_t
        {
            const parameter_t&  m_parameter;
        };

        struct enum_t
        {
            string_t        m_value;                        ///< the stored value as string
            strings_t       m_domain;                       ///< domain of available values as strings
        };

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        struct range_t
        {
            tscalar         m_value{0};                     ///< the stored value
            tscalar         m_min{0}, m_max{0};             ///< the allowed [min, max] range
            LEorLT          m_mincomp, m_maxcomp;           ///<
        };

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        struct pair_range_t
        {
            tscalar         m_value1{0}, m_value2{0};       ///< the stored values
            tscalar         m_min{0}, m_max{0};             ///< the allowed [min, max] range
            LEorLT          m_mincomp, m_valcomp, m_maxcomp;///<
        };

        using irange_t = range_t<int64_t>;
        using frange_t = range_t<scalar_t>;
        using iprange_t = pair_range_t<int64_t>;
        using fprange_t = pair_range_t<scalar_t>;

        using storage_t = std::variant<std::monostate, enum_t, irange_t, frange_t, iprange_t, fprange_t>;

        ///
        /// \brief default constructor
        ///
        parameter_t();

        ///
        /// \brief return a constrained enumeration parameter.
        ///
        template
        <
            typename tenum,
            std::enable_if_t<std::is_enum_v<tenum>, bool> = true
        >
        static parameter_t make_enum(string_t name, tenum value)
        {
            return make_enum_(std::move(name), value);
        }

        ///
        /// \brief return a floating point parameter constrained to the given range:
        ///     min LE/LT value LE/LT max.
        ///
        template
        <
            typename tmin, typename tvalue, typename tmax,
            std::enable_if_t<std::is_arithmetic_v<tmin>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tvalue>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tmax>, bool> = true
        >
        static parameter_t make_float(string_t name,
            tmin min, LEorLT mincomp, tvalue value, LEorLT maxcomp, tmax max)
        {
            return make_scalar<scalar_t>(std::move(name), min, mincomp, value, maxcomp, max);
        }

        ///
        /// \brief return an integer parameter constrained to the given range:
        ///     min LE/LT value LE/LT max.
        ///
        template
        <
            typename tmin, typename tvalue, typename tmax,
            std::enable_if_t<std::is_arithmetic_v<tmin>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tvalue>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tmax>, bool> = true
        >
        static parameter_t make_integer(string_t name,
            tmin min, LEorLT mincomp, tvalue value, LEorLT maxcomp, tmax max)
        {
            return make_scalar<int64_t>(std::move(name), min, mincomp, value, maxcomp, max);
        }

        ///
        /// \brief return an ordered floating point pair parameter constrained to the given range:
        ///     min LE/LT value1 LE/LT value2 LE/LT max.
        ///
        template
        <
            typename tmin, typename tvalue1, typename tvalue2, typename tmax,
            std::enable_if_t<std::is_arithmetic_v<tmin>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tvalue1>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tvalue2>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tmax>, bool> = true
        >
        static parameter_t make_float_pair(string_t name,
            tmin min, LEorLT mincomp, tvalue1 value1, LEorLT valcomp, tvalue2 value2, LEorLT maxcomp, tmax max)
        {
            return make_scalar<scalar_t>(std::move(name), min, mincomp, value1, valcomp, value2, maxcomp, max);
        }

        ///
        /// \brief return an integer pair parameter constrained to the given range:
        ///     min LE/LT value1 LE/LT value2 LE/LT max.
        ///
        template
        <
            typename tmin, typename tvalue1, typename tvalue2, typename tmax,
            std::enable_if_t<std::is_arithmetic_v<tmin>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tvalue1>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tvalue2>, bool> = true,
            std::enable_if_t<std::is_arithmetic_v<tmax>, bool> = true
        >
        static parameter_t make_integer_pair(string_t name,
            tmin min, LEorLT mincomp, tvalue1 value1, LEorLT valcomp, tvalue2 value2, LEorLT maxcomp, tmax max)
        {
            return make_scalar<int64_t>(std::move(name), min, mincomp, value1, valcomp, value2, maxcomp, max);
        }

        ///
        /// \brief change the parameter's value, throws an exception if not possible.
        ///
        parameter_t& operator=(int32_t);
        parameter_t& operator=(int64_t);
        parameter_t& operator=(scalar_t);
        parameter_t& operator=(string_t);
        parameter_t& operator=(std::tuple<int32_t, int32_t>);
        parameter_t& operator=(std::tuple<int64_t, int64_t>);
        parameter_t& operator=(std::tuple<scalar_t, scalar_t>);

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        parameter_t& operator=(tenum value)
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
        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        tenum value() const
        {
            if (const auto* param = std::get_if<enum_t>(&m_storage))
            {
                return from_string<tenum>(param->m_value);
            }
            logical_error();
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        tscalar value() const
        {
            if (const auto* param = std::get_if<irange_t>(&m_storage))
            {
                return static_cast<tscalar>(param->m_value);
            }
            else if (const auto* param = std::get_if<frange_t>(&m_storage))
            {
                return static_cast<tscalar>(param->m_value);
            }
            logical_error();
        }

        template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
        std::tuple<tscalar, tscalar> value_pair() const
        {
            if (const auto* param = std::get_if<iprange_t>(&m_storage))
            {
                return std::make_tuple(static_cast<tscalar>(param->m_value1), static_cast<tscalar>(param->m_value2));
            }
            else if (const auto* param = std::get_if<fprange_t>(&m_storage))
            {
                return std::make_tuple(static_cast<tscalar>(param->m_value1), static_cast<tscalar>(param->m_value2));
            }
            logical_error();
        }

        ///
        /// \brief serialize from the given binary stream.
        ///
        /// NB: any error is considered critical and expected to result in an exception.
        ///
        void read(std::istream&);

        ///
        /// \brief serialize to the given binary stream.
        ///
        /// NB: any error is considered critical and expected to result in an exception.
        ///
        void write(std::ostream&) const;

        ///
        /// \brief access functions.
        ///
        const auto& name() const { return m_name; }
        const auto& storage() const { return m_storage; }
        auto value() const { return value_t{*this}; }
        auto domain() const { return domain_t{*this}; }

    private:

        parameter_t(string_t name, enum_t);
        parameter_t(string_t name, irange_t);
        parameter_t(string_t name, frange_t);
        parameter_t(string_t name, iprange_t);
        parameter_t(string_t name, fprange_t);

        [[noreturn]] void logical_error() const;

        template <typename tenum>
        static parameter_t make_enum_(string_t name, tenum value)
        {
            static const auto options = enum_string<tenum>();

            strings_t domain{options.size()};
            std::transform(options.begin(), options.end(), domain.begin(), [] (const auto& v) { return v.second; });

            return parameter_t
            {
                std::move(name),
                enum_t{scat(value), std::move(domain)}
            };
        }

        template <typename tscalar, typename tmin, typename tvalue, typename tmax>
        static parameter_t make_scalar(string_t name,
            tmin min, LEorLT mincomp, tvalue value, LEorLT maxcomp, tmax max)
        {
            return parameter_t
            {
                std::move(name),
                range_t<tscalar>
                {
                    static_cast<tscalar>(value),
                    static_cast<tscalar>(min), static_cast<tscalar>(max),
                    mincomp, maxcomp
                }
            };
        }

        template <typename tscalar, typename tmin, typename tvalue1, typename tvalue2, typename tmax>
        static parameter_t make_scalar(string_t name,
            tmin min, LEorLT mincomp, tvalue1 value1, LEorLT valcomp, tvalue2 value2, LEorLT maxcomp, tmax max)
        {
            return parameter_t
            {
                std::move(name),
                pair_range_t<tscalar>
                {
                    static_cast<tscalar>(value1), static_cast<tscalar>(value2),
                    static_cast<tscalar>(min), static_cast<tscalar>(max),
                    mincomp, valcomp, maxcomp
                }
            };
        }

        // attributes
        string_t        m_name;         ///<
        storage_t       m_storage;      ///<
    };

    using parameters_t = std::vector<parameter_t>;

    NANO_PUBLIC bool operator==(const parameter_t&, const parameter_t&);
    NANO_PUBLIC bool operator!=(const parameter_t&, const parameter_t&);
    NANO_PUBLIC std::ostream& operator<<(std::ostream&, const parameter_t&);
    NANO_PUBLIC std::ostream& operator<<(std::ostream&, const parameter_t::value_t&);
    NANO_PUBLIC std::ostream& operator<<(std::ostream&, const parameter_t::domain_t&);

    NANO_PUBLIC std::istream& read(std::istream& stream, parameter_t&);
    NANO_PUBLIC std::ostream& write(std::ostream& stream, const parameter_t&);
}
