#pragma once

#include <cmath>
#include <variant>
#include <nano/scalar.h>
#include <nano/core/logger.h>
#include <nano/core/strutil.h>

namespace nano
{
    ///
    /// \brief less or equal operator.
    ///
    struct LE_t
    {
        static auto name()
        {
            return " <= ";
        }

        template <typename tscalar>
        static auto check(tscalar value1, tscalar value2)
        {
            return value1 <= value2;
        }
    };

    ///
    /// \brief less than operator.
    ///
    struct LT_t
    {
        static auto name()
        {
            return " < ";
        }

        template <typename tscalar>
        static auto check(tscalar value1, tscalar value2)
        {
            return value1 < value2;
        }
    };

    static const auto LE = LE_t{};
    static const auto LT = LT_t{};

    using LEorLT = std::variant<LE_t, LT_t>;

    namespace detail
    {
        inline auto name(const LEorLT& lelt)
        {
            return std::holds_alternative<LE_t>(lelt) ? LE_t::name() : LT_t::name();
        }

        template <typename tscalar>
        inline auto check(const LEorLT& lelt, tscalar value1, tscalar value2)
        {
            return std::holds_alternative<LE_t>(lelt) ? LE_t::check(value1, value2) : LT_t::check(value1, value2);
        }
    }

    ///
    /// \brief stores a scalar parameter and enforces its value to be within the given range:
    ///     min LE/LT value LE/LT max.
    ///
    template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
    class param1_t
    {
    public:

        param1_t() = default;

        template <typename tmin, typename tmincomp, typename tvalue, typename tmaxcomp, typename tmax>
        param1_t(string_t name, tmin min, tmincomp mincomp, tvalue value, tmaxcomp maxcomp, tmax max) :
            m_name(std::move(name)),
            m_min(static_cast<tscalar>(min)),
            m_max(static_cast<tscalar>(max)),
            m_mincomp(mincomp),
            m_maxcomp(maxcomp)
        {
            set(value);
        }

        template <typename tvalue>
        param1_t& operator=(tvalue value)
        {
            set(value);
            return *this;
        }

        template <typename tvalue>
        void set(tvalue value)
        {
            m_value = checked(value);
        }

        auto min() const { return m_min; }
        auto max() const { return m_max; }
        auto get() const { return m_value; }
        const auto& name() const { return m_name; }
        auto minLE() const { return std::holds_alternative<LE_t>(m_mincomp); }
        auto maxLE() const { return std::holds_alternative<LE_t>(m_maxcomp); }

    private:

        template <typename tvalue>
        auto checked(tvalue _value) const
        {
            const auto value = static_cast<tscalar>(_value);

            critical(
                !std::isfinite(_value) || !std::isfinite(value) ||
                !detail::check(m_mincomp, m_min, value) ||
                !detail::check(m_maxcomp, value, m_max),
                "invalid parameter '", m_name, "': !(", m_min, detail::name(m_mincomp),
                value, detail::name(m_maxcomp), m_max, ")");

            return value;
        }

        // attributes
        string_t        m_name;                         ///< parameter name
        tscalar         m_value{0};                     ///< the stored value
        tscalar         m_min{0}, m_max{0};             ///< the allowed [min, max] range
        LEorLT          m_mincomp, m_maxcomp;           ///<
    };

    ///
    /// \brief stores two ordered scalar parameters and enforces their values to be within the given range:
    ///     min LE/LT value1 LE/LT value2 LE/LT max
    ///
    template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
    class param2_t
    {
    public:

        template
        <
            typename tmin, typename tmincomp,
            typename tvalue1, typename tvalcomp, typename tvalue2,
            typename tmaxcomp, typename tmax
        >
        param2_t(
            string_t name,
            tmin min, tmincomp mincomp, tvalue1 value1, tvalcomp valcomp, tvalue2 value2, tmaxcomp maxcomp, tmax max) :
            m_name(std::move(name)),
            m_min(static_cast<tscalar>(min)),
            m_max(static_cast<tscalar>(max)),
            m_mincomp(mincomp),
            m_valcomp(valcomp),
            m_maxcomp(maxcomp)
        {
            set(value1, value2);
        }

        template <typename tvalue1, typename tvalue2>
        void set(tvalue1 value1, tvalue2 value2)
        {
            std::tie(m_value1, m_value2) = checked(value1, value2);
        }

        auto min() const { return m_min; }
        auto max() const { return m_max; }
        auto get1() const { return m_value1; }
        auto get2() const { return m_value2; }
        const auto& name() const { return m_name; }

    private:

        template <typename tvalue1, typename tvalue2>
        auto checked(tvalue1 _value1, tvalue2 _value2) const
        {
            const auto value1 = static_cast<tscalar>(_value1);
            const auto value2 = static_cast<tscalar>(_value2);

            critical(
                !std::isfinite(_value1) || !std::isfinite(value1) ||
                !std::isfinite(_value2) || !std::isfinite(value2) ||
                !detail::check(m_mincomp, m_min, value1) ||
                !detail::check(m_valcomp, value1, value2) ||
                !detail::check(m_maxcomp, value2, m_max),
                "invalid parameter '", m_name, "': !(",
                m_min, detail::name(m_mincomp),
                value1, detail::name(m_valcomp),
                value2, detail::name(m_maxcomp), m_max, ")");

            return std::make_pair(value1, value2);
        }

        // attributes
        string_t        m_name;                             ///< parameter name
        tscalar         m_value1, m_value2;                 ///< the stored values
        tscalar         m_min, m_max;                       ///< the allowed [min, max] range
        LEorLT          m_mincomp, m_valcomp, m_maxcomp;    ///<
    };

    using iparam1_t = param1_t<int64_t>;
    using uparam1_t = param1_t<uint64_t>;
    using sparam1_t = param1_t<scalar_t>;

    using iparam2_t = param2_t<int64_t>;
    using uparam2_t = param2_t<uint64_t>;
    using sparam2_t = param2_t<scalar_t>;

    ///
    /// \brief stores an enumeration parameter and enforces its value to be valid.
    ///
    class NANO_PUBLIC eparam1_t
    {
    public:

        eparam1_t();

        eparam1_t(string_t name, string_t value, strings_t domain);

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        eparam1_t(string_t name, tenum value) :
            m_name(std::move(name))
        {
            set(value);
        }

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        eparam1_t& operator=(tenum value)
        {
            set(value);
            return *this;
        }

        void set(string_t value);

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        void set(tenum value)
        {
            update<tenum>();
            set(scat(value));
        }

        const auto& name() const { return m_name; }
        const auto& get() const { return m_value; }
        const auto& domain() const { return m_domain; }

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        auto as() const
        {
            return from_string<tenum>(m_value);
        }

    private:

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        void update()
        {
            static const auto options = enum_string<tenum>();
            const auto* tenum_name = typeid(tenum).name();
            if (m_typeid_name != tenum_name)
            {
                m_typeid_name = tenum_name;
                m_domain.resize(options.size());
                std::transform(
                    options.begin(), options.end(), m_domain.begin(),
                    [] (const auto& v) { return v.second; });
            }
        }

        // attributes
        string_t        m_name;                         ///< parameter name
        string_t        m_value;                        ///< the stored value as string
        strings_t       m_domain;                       ///< domain of available values as strings
        string_t        m_typeid_name;
    };

    ///
    /// \brief serializable parameter addressable by name.
    ///
    /// NB: the parameter can be an integer, a scalar or an enumeration and
    ///     it can be serialized to and from binary streams.
    ///
    class NANO_PUBLIC parameter_t
    {
    public:

        using storage_t = std::variant<eparam1_t, iparam1_t, sparam1_t>;

        ///
        /// \brief default constructor
        ///
        parameter_t() = default;

        ///
        /// \brief constructor
        ///
        explicit parameter_t(eparam1_t);
        explicit parameter_t(iparam1_t);
        explicit parameter_t(sparam1_t);

        ///
        /// \brief change the parameter's value.
        ///
        void set(int32_t);
        void set(int64_t);
        void set(scalar_t);

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        void set(tenum value)
        {
            eparam().set(value);
        }

        ///
        /// \brief retrieve the current parameter's value.
        ///
        int64_t ivalue() const;
        scalar_t svalue() const;

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        tenum evalue() const
        {
            return eparam().as<tenum>();
        }

        ///
        /// \brief returns true if the parameter is an enumeration, an integer or a scalar.
        ///
        bool is_evalue() const;
        bool is_ivalue() const;
        bool is_svalue() const;

        ///
        /// \brief returns the parameter's name if initialized, otherwise throws an exception.
        ///
        const string_t& name() const;

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
        /// \brief returns the stored parameters.
        ///
        const eparam1_t& eparam() const;
        const iparam1_t& iparam() const;
        const sparam1_t& sparam() const;

    private:

        eparam1_t& eparam();
        iparam1_t& iparam();
        sparam1_t& sparam();

        // attributes
        storage_t       m_storage;      ///<
    };

    using parameters_t = std::vector<parameter_t>;

    NANO_PUBLIC std::ostream& operator<<(std::ostream&, const parameter_t&);

    NANO_PUBLIC std::istream& read(std::istream& stream, parameter_t&);
    NANO_PUBLIC std::ostream& write(std::ostream& stream, const parameter_t&);
}
