#pragma once

#include <cmath>
#include <nano/scalar.h>
#include <nano/string.h>

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
        static auto check(const tscalar value1, const tscalar value2)
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
        static auto check(const tscalar value1, const tscalar value2)
        {
            return value1 < value2;
        }
    };

    static const auto LE = LE_t{};
    static const auto LT = LT_t{};

    ///
    /// \brief TODO: replace this with std::variant<LE, LT> when moving to C++17!
    ///
    class LEorLT
    {
    public:

        LEorLT() = default;
        explicit LEorLT(LE_t) {}
        explicit LEorLT(LT_t) : m_LE(false) {}

        [[nodiscard]] auto name() const
        {
            return m_LE ? LE_t::name() : LT_t::name();
        }

        template <typename tscalar>
        [[nodiscard]] auto check(const tscalar value1, const tscalar value2) const
        {
            return m_LE ? LE_t::check(value1, value2) : LT_t::check(value1, value2);
        }

    private:

        // attributes
        bool    m_LE{true};     ///<
    };

    ///
    /// \brief stores a scalar parameter and enforces its value to be within the given range:
    ///     min LE/LT value LE/LT max.
    ///
    template <typename tscalar>
    class param1_t
    {
    public:

        template <typename tmin, typename tmincomp, typename tvalue, typename tmaxcomp, typename tmax>
        param1_t(const char* name, tmin min, tmincomp mincomp, tvalue value, tmaxcomp maxcomp, tmax max) :
            m_name(name),
            m_min(static_cast<tscalar>(min)),
            m_max(static_cast<tscalar>(max)),
            m_mincomp(mincomp),
            m_maxcomp(maxcomp)
        {
            set(value);
        }

        template <typename tvalue>
        param1_t& operator=(const tvalue value)
        {
            set(value);
            return *this;
        }

        template <typename tvalue>
        void set(const tvalue value)
        {
            m_value = checked(value);
        }

        [[nodiscard]] auto min() const { return m_min; }
        [[nodiscard]] auto max() const { return m_max; }
        [[nodiscard]] auto get() const { return m_value; }

    private:

        template <typename tvalue>
        [[nodiscard]] auto checked(const tvalue _value) const
        {
            const auto value = static_cast<tscalar>(_value);
            if (    !std::isfinite(_value) || !std::isfinite(value) ||
                    !m_mincomp.check(m_min, value) ||
                    !m_maxcomp.check(value, m_max))
            {
                throw std::invalid_argument(scat("invalid parameter '", m_name, "': !(",
                    m_min, m_mincomp.name(), value, m_maxcomp.name(), m_max, ")"));
            }

            return value;
        }

        // attributes
        const char*     m_name{nullptr};                ///< parameter name
        tscalar         m_value;                        ///< the stored value
        tscalar         m_min, m_max;                   ///< the allowed [min, max] range
        LEorLT          m_mincomp, m_maxcomp;           ///<
    };

    ///
    /// \brief stores two ordered scalar parameters and enforces their values to be within the given range:
    ///     min LE/LT value1 LE/LT value2 LE/LT max
    ///
    template <typename tscalar>
    class param2_t
    {
    public:

        template <typename tmin, typename tmincomp, typename tvalue1, typename tvalcomp, typename tvalue2, typename tmaxcomp, typename tmax>
        param2_t(const char* name, tmin min, tmincomp mincomp, tvalue1 value1, tvalcomp valcomp, tvalue2 value2, tmaxcomp maxcomp, tmax max) :
            m_name(name),
            m_min(static_cast<tscalar>(min)),
            m_max(static_cast<tscalar>(max)),
            m_mincomp(mincomp),
            m_valcomp(valcomp),
            m_maxcomp(maxcomp)
        {
            set(value1, value2);
        }

        template <typename tvalue1, typename tvalue2>
        void set(const tvalue1 value1, const tvalue2 value2)
        {
            std::tie(m_value1, m_value2) = checked(value1, value2);
        }

        [[nodiscard]] auto min() const { return m_min; }
        [[nodiscard]] auto max() const { return m_max; }
        [[nodiscard]] auto get1() const { return m_value1; }
        [[nodiscard]] auto get2() const { return m_value2; }

    private:

        template <typename tvalue1, typename tvalue2>
        [[nodiscard]] auto checked(const tvalue1 _value1, const tvalue2 _value2) const
        {
            const auto value1 = static_cast<tscalar>(_value1);
            const auto value2 = static_cast<tscalar>(_value2);
            if (    !std::isfinite(_value1) || !std::isfinite(value1) ||
                    !std::isfinite(_value2) || !std::isfinite(value2) ||
                    !m_mincomp.check(m_min, value1) ||
                    !m_valcomp.check(value1, value2) ||
                    !m_maxcomp.check(value2, m_max))
            {
                throw std::invalid_argument(scat("invalid parameter '", m_name, "': !(",
                    m_min, m_mincomp.name(), value1, m_valcomp.name(), value2, m_maxcomp.name(), m_max, ")"));
            }

            return std::make_pair(value1, value2);
        }

        // attributes
        const char*     m_name{nullptr};                    ///< parameter name
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
}
