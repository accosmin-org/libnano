#pragma once

#include <nano/tensor/tensor.h>

namespace nano
{
///
/// \brief pretty-print the given tensor.
///
template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
std::ostream& pprint(std::ostream& stream, const tensor_t<tstorage, tscalar, trank>& tensor,
                     const tensor_size_t prefix_space = 0, const tensor_size_t prefix_delim = 0,
                     const tensor_size_t suffix = 0)
{
    [[maybe_unused]] const auto sprint = [&](const char c, const tensor_size_t count) -> std::ostream&
    {
        for (tensor_size_t i = 0; i < count; ++i)
        {
            stream << c;
        }
        return stream;
    };

    if (prefix_space == 0 && prefix_delim == 0 && suffix == 0)
    {
        stream << "shape: " << tensor.dims() << "\n";
    }

    if constexpr (trank == 1)
    {
        sprint('[', prefix_delim + 1);
        if constexpr (std::is_same_v<tscalar, int8_t>)
        {
            stream << tensor.vector().transpose().template cast<int16_t>();
        }
        else if constexpr (std::is_same_v<tscalar, uint8_t>)
        {
            stream << tensor.vector().transpose().template cast<uint16_t>();
        }
        else
        {
            stream << tensor.vector().transpose();
        }
        sprint(']', suffix + 1);
    }
    else if constexpr (trank == 2)
    {
        const auto matrix = tensor.matrix();
        for (tensor_size_t row = 0, nrows = matrix.rows(); row < nrows; ++row)
        {
            if (row == 0)
            {
                sprint(' ', prefix_space);
                sprint('[', prefix_delim + 2);
            }
            else
            {
                sprint(' ', prefix_space + prefix_delim + 1);
                sprint('[', 1);
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
                sprint(']', suffix + 2);
            }
        }
    }
    else
    {
        for (tensor_size_t row = 0, nrows = tensor.template size<0>(); row < nrows; ++row)
        {
            print(stream, tensor.tensor(row), (row == 0) ? prefix_space : (prefix_space + prefix_delim + 1),
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
