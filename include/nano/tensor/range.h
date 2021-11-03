#pragma once

#include <nano/tensor/index.h>

namespace nano
{
    ///
    /// \brief continuous range of dimensions [begin, end).
    ///
    class tensor_range_t
    {
    public:

        tensor_range_t() = default;

        tensor_range_t(const tensor_size_t begin, const tensor_size_t end) :
            m_begin(begin),
            m_end(end)
        {
        }

        auto begin() const
        {
            return m_begin;
        }

        auto end() const
        {
            return m_end;
        }

        auto size() const
        {
            return end() - begin();
        }

        ///
        /// \brief check if a range is valid, so that [begin, end) is included in [0, size).
        ///
        auto valid(const tensor_size_t size) const
        {
            return 0 <= m_begin && m_begin < m_end && m_end <= size;
        }

    private:

        // attributes
        tensor_size_t       m_begin{0};     ///<
        tensor_size_t       m_end{0};       ///<
    };

    ///
    /// \brief creates a range of dimensions.
    ///
    inline auto make_range(const tensor_size_t begin, const tensor_size_t end)
    {
        return tensor_range_t{begin, end};
    }
}
