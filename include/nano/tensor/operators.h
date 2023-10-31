#pragma once

#include <nano/tensor/pprint.h>

namespace nano
{
///
/// \brief pretty-print the given tensor.
///
template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
std::ostream& operator<<(std::ostream& stream, const tensor_t<tstorage, tscalar, trank>& tensor)
{
    return pprint(stream, tensor);
}

///
/// \brief compare two tensors element-wise.
///
template <template <typename, size_t> class tstorage1, template <typename, size_t> class tstorage2, typename tscalar,
          size_t trank>
bool operator==(const tensor_t<tstorage1, tscalar, trank>& lhs, const tensor_t<tstorage2, tscalar, trank>& rhs)
{
    return lhs.dims() == rhs.dims() && lhs.vector() == rhs.vector();
}

///
/// \brief compare two tensors element-wise.
///
template <template <typename, size_t> class tstorage1, template <typename, size_t> class tstorage2, typename tscalar,
          size_t trank>
bool operator!=(const tensor_t<tstorage1, tscalar, trank>& lhs, const tensor_t<tstorage2, tscalar, trank>& rhs)
{
    return lhs.dims() != rhs.dims() || lhs.vector() != rhs.vector();
}
} // namespace nano
