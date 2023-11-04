#pragma once

#include <nano/tensor/index.h>
#include <ostream>

namespace nano
{
template <template <typename, size_t> class, typename, size_t>
class tensor_t;

namespace detail
{
inline void sprint(std::ostream& stream, const char c, const tensor_size_t count)
{
    for (tensor_size_t i = 0; i < count; ++i)
    {
        stream << c;
    }
};
} // namespace detail

///
/// \brief pretty-print the given tensor.
///
template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
std::ostream& pprint(std::ostream& stream, const tensor_t<tstorage, tscalar, trank>& tensor,
                     const tensor_size_t prefix_space = 0, const tensor_size_t prefix_delim = 0,
                     const tensor_size_t suffix = 0)
{
    if (prefix_space == 0 && prefix_delim == 0 && suffix == 0)
    {
        stream << "shape: " << tensor.dims() << "\n";
    }

    if constexpr (trank == 1)
    {
        detail::sprint(stream, '[', prefix_delim + 1);
        if constexpr (std::is_same_v<tscalar, int8_t>)
        {
            stream << tensor.transpose().template cast<int16_t>();
        }
        else if constexpr (std::is_same_v<tscalar, uint8_t>)
        {
            stream << tensor.transpose().template cast<uint16_t>();
        }
        else
        {
            stream << tensor.vector().transpose();
        }
        detail::sprint(stream, ']', suffix + 1);
    }
    else if constexpr (trank == 2)
    {
        const auto matrix = tensor.matrix();
        for (tensor_size_t row = 0, nrows = matrix.rows(); row < nrows; ++row)
        {
            if (row == 0)
            {
                detail::sprint(stream, ' ', prefix_space);
                detail::sprint(stream, '[', prefix_delim + 2);
            }
            else
            {
                detail::sprint(stream, ' ', prefix_space + prefix_delim + 1);
                detail::sprint(stream, '[', 1);
            }

            if constexpr (std::is_same_v<tscalar, int8_t>)
            {
                stream << matrix.row(row).template cast<int16_t>();
            }
            else if constexpr (std::is_same_v<tscalar, uint8_t>)
            {
                stream << matrix.row(row).template cast<uint16_t>();
            }
            else
            {
                stream << matrix.row(row);
            }

            if (row + 1 < nrows)
            {
                stream << "]\n";
            }
            else
            {
                detail::sprint(stream, ']', suffix + 2);
            }
        }
    }
    else
    {
        for (tensor_size_t row = 0, nrows = tensor.template size<0>(); row < nrows; ++row)
        {
            pprint(stream, tensor.tensor(row), (row == 0) ? prefix_space : (prefix_space + prefix_delim + 1),
                   (row == 0) ? (prefix_delim + 1) : 0, suffix);

            if (row + 1 < nrows)
            {
                stream << "\n";
            }
            else
            {
                stream << "]";
            }
        }
    }
    return stream;
}
} // namespace nano
