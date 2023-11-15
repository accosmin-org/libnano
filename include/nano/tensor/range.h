#pragma once

#include <nano/tensor/index.h>

namespace nano
{
///
/// \brief continuous range [begin, end) of indices.
///
class tensor_range_t
{
public:
    ///
    /// \brief constructor
    ///
    tensor_range_t() = default; // LCOV_EXCL_LINE

    ///
    /// \brief constructor
    ///
    tensor_range_t(const tensor_size_t begin, const tensor_size_t end)
        : m_begin(begin)
        , m_end(end)
    {
    }

    ///
    /// \brief returns the begining of the range.
    ///
    tensor_size_t begin() const { return m_begin; }

    ///
    /// \brief returns the end of the range.
    ///
    tensor_size_t end() const { return m_end; }

    ///
    /// \brief returns the size of the range.
    ///
    tensor_size_t size() const { return m_end - m_begin; }

    ///
    /// \brief check if a range is valid, so that [begin, end) is included in [0, size).
    ///
    bool valid(const tensor_size_t size) const { return 0 <= m_begin && m_begin < m_end && m_end <= size; }

private:
    // attributes
    tensor_size_t m_begin{0}; ///<
    tensor_size_t m_end{0};   ///<
};

///
/// \brief creates a range of dimensions.
///
inline auto make_range(tensor_size_t begin, tensor_size_t end)
{
    return tensor_range_t{begin, end};
}
} // namespace nano
