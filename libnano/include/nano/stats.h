#pragma once

#include <cmath>
#include <limits>
#include <vector>
#include <cassert>
#include <numeric>
#include <ostream>
#include <algorithm>

namespace nano
{
    ///
    /// \brief collects numerical values and computes statistics like:
    ///     - average, standard deviation
    ///     - minimum and maximum values
    ///     - median, percentile etc.
    ///
    class stats_t
    {
    public:

        using value_t = double;
        using values_t = std::vector<value_t>;

        ///
        /// \brief constructor
        ///
        stats_t() = default;

        ///
        /// \brief update statistics with new values
        ///
        template <typename tscalar>
        void operator()(const tscalar value)
        {
            m_values.push_back(static_cast<value_t>(value));
        }

        template <typename tscalar, typename... tscalars>
        void operator()(const tscalar value, const tscalars... values)
        {
            operator()(value);
            operator()(values...);
        }

        ///
        /// \brief merge statistics
        ///
        void operator()(const stats_t& other)
        {
            m_values.insert(m_values.end(), other.m_values.begin(), other.m_values.end());
        }

        ///
        /// \brief update statistics with the given [begin, end) range
        ///
        template <typename titerator, typename = typename std::iterator_traits<titerator>::value_type>
        void operator()(titerator begin, const titerator end)
        {
            for ( ; begin != end; ++ begin)
            {
                operator()(*begin);
            }
        }

        ///
        /// \brief reset statistics
        ///
        void clear()
        {
            m_values.clear();
        }

        ///
        /// \brief returns the number of values
        ///
        auto count() const
        {
            return m_values.size();
        }

        ///
        /// \brief returns the minimum
        ///
        auto min() const
        {
            assert(!m_values.empty());
            return *std::min_element(m_values.begin(), m_values.end());
        }

        ///
        /// \brief returns the maximum
        ///
        auto max() const
        {
            assert(!m_values.empty());
            return *std::max_element(m_values.begin(), m_values.end());
        }

        ///
        /// \brief returns the sum
        ///
        auto sum1() const
        {
            return  std::accumulate(m_values.begin(), m_values.end(), value_t(0),
                    [] (const auto s, const auto v) { return s + v; });
        }

        ///
        /// \brief returns the sum of squares
        ///
        auto sum2() const
        {
            return  std::accumulate(m_values.begin(), m_values.end(), value_t(0),
                    [] (const auto s, const auto v) { return s + v * v; });
        }

        ///
        /// \brief returns the average
        ///
        auto avg() const
        {
            assert(count() > 0);
            return sum1() / static_cast<value_t>(count());
        }

        ///
        /// \brief returns the variance
        ///
        auto var() const
        {
            assert(count() > 0);
            return sum2() / static_cast<value_t>(count()) - avg() * avg();
        }

        ///
        /// \brief returns the population standard deviation
        ///
        auto stdev() const
        {
            return std::sqrt(var());
        }

        ///
        /// \brief returns the percentile
        ///
        auto percentile(const size_t percentage)
        {
            assert(count() > 0);
            assert(percentage > 0 && percentage < 100);
            const auto pos = m_values.size() * percentage / 100;
            std::nth_element(m_values.begin(), m_values.begin() + pos, m_values.end());
            return m_values[pos];
        }

        ///
        /// \brief returns the median
        ///
        auto median()
        {
            return percentile(50);
        }

        ///
        /// \brief check if valid (enough values collected)
        ///
        operator bool() const { return count() > 1; }

    private:

        // attributes
        values_t    m_values;       ///< store all values
    };

    inline std::ostream& operator<<(std::ostream& os, const stats_t& stats)
    {
        return  !stats ?
                os :
                (os << stats.avg() << "+/-" << stats.stdev() << "[" << stats.min() << "," << stats.max() << "]");
    }
}
