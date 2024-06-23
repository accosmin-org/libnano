#pragma once

#include <nano/tensor.h>
#include <numeric>

namespace nano
{
///
/// \brief generate all combinations of the given number of elements per dimension
///     (e.g. number of distinct values per parameter).
///
template <class tindex, std::enable_if_t<std::is_integral_v<tindex>, bool> = true>
class combinatorial_iterator_t
{
public:
    explicit combinatorial_iterator_t(const tensor_mem_t<tindex, 1>& counts)
        : m_counts(counts)
        , m_current(counts.size())
        , m_dimensions(counts.size())
        , m_combinations(product(counts))
    {
        m_current.zero();

        assert(m_dimensions > 0);
        assert(m_combinations > 0);
    }

    explicit operator bool() const { return m_combination < m_combinations; }

    combinatorial_iterator_t& operator++()
    {
        while (m_combination < m_combinations)
        {
            if (m_dimension + 1 == m_dimensions)
            {
                if (m_current(m_dimension) + 1 < m_counts(m_dimension))
                {
                    ++m_combination;
                    ++m_current(m_dimension);
                    return *this;
                }

                while (m_dimension >= 0)
                {
                    if (m_current(m_dimension) + 1 >= m_counts(m_dimension))
                    {
                        m_current(m_dimension) = 0;
                        --m_dimension;
                    }
                    else
                    {
                        ++m_combination;
                        ++m_current(m_dimension);
                        return *this;
                    }
                }
            }
            else
            {
                ++m_dimension;
                m_current(m_dimension) = 0;
            }
        }

        return *this;
    }

    const tensor_mem_t<tindex, 1>& operator*() const { return m_current; }

    tensor_size_t index() const { return m_combination; }

    tensor_size_t size() const { return m_combinations; }

private:
    static tensor_size_t product(const tensor_mem_t<tindex, 1>& counts)
    {
        const auto multiply = [](tensor_size_t acc, tindex val) { return acc * static_cast<tensor_size_t>(val); };
        return std::accumulate(std::begin(counts), std::end(counts), tensor_size_t{1}, multiply);
    }

    // attributes
    tensor_mem_t<tindex, 1> m_counts;          ///<
    tensor_mem_t<tindex, 1> m_current;         ///< current combination as indices in the counts
    tensor_size_t           m_dimension{0};    ///< index of the current dimension
    tensor_size_t           m_dimensions{0};   ///< total number of dimensions
    tensor_size_t           m_combination{0};  ///< index of the current combination
    tensor_size_t           m_combinations{1}; ///< total number of combinations
};
} // namespace nano
