#pragma once

#include <array>
#include <vector>
#include <cassert>
#include <ostream>
#include <eigen3/Eigen/Core>

namespace nano
{
    using tensor_size_t = Eigen::Index;

    ///
    /// \brief dimensions of a multi-dimensional tensor.
    ///
    template <size_t trank>
    using tensor_dims_t = std::array<tensor_size_t, trank>;

    ///
    /// \brief stores the dimensions of a tensor.
    ///
    template <typename... tsizes>
    auto make_dims(const tsizes... sizes)
    {
        return tensor_dims_t<sizeof...(sizes)>({{sizes...}});
    }

    template <size_t trank>
    auto cat_dims(const tensor_size_t size, const tensor_dims_t<trank>& dims)
    {
        tensor_dims_t<trank + 1> xdims{{size}};
        std::get<0>(xdims) = size;
        std::copy(dims.cbegin(), dims.cend(), xdims.begin() + 1);
        return xdims;
    }

    namespace detail
    {
        // todo: simplify this part with C++17 using "if constexpr"
        template <size_t idim, size_t trank>
        struct product_t
        {
            static tensor_size_t get(const tensor_dims_t<trank>& dims)
            {
                return std::get<idim>(dims) * product_t<idim + 1, trank>::get(dims);
            }
        };

        template <size_t trank>
        struct product_t<trank, trank>
        {
            static tensor_size_t get(const tensor_dims_t<trank>&)
            {
                return 1;
            }
        };

        template <size_t idim, size_t trank>
        tensor_size_t product(const tensor_dims_t<trank>& dims)
        {
            return product_t<idim, trank>::get(dims);
        }

        template <size_t idim, size_t trank>
        tensor_size_t get_index(const tensor_dims_t<trank>&, const tensor_size_t index)
        {
            return index;
        }

        template <size_t idim, size_t trank, typename... tindices>
        tensor_size_t get_index(const tensor_dims_t<trank>& dims,
            const tensor_size_t index, const tindices... indices)
        {
            assert(index >= 0 && index < std::get<idim>(dims));
            return index * product<idim + 1>(dims) + get_index<idim + 1>(dims, indices...);
        }

        template <size_t idim, size_t trank>
        tensor_size_t get_index0(const tensor_dims_t<trank>&)
        {
                return 0;
        }

        template <size_t idim, size_t trank, typename... tindices>
        tensor_size_t get_index0(const tensor_dims_t<trank>& dims,
            const tensor_size_t index, const tindices... indices)
        {
            assert(index >= 0 && index < std::get<idim>(dims));
            return index * product<idim + 1>(dims) + get_index0<idim + 1>(dims, indices...);
        }

        template <size_t idim, size_t trank>
        struct dims_t
        {
            template <size_t trankx>
            static void set(const tensor_dims_t<trank>& dims, tensor_dims_t<trankx>& dimsx)
            {
                static_assert(idim >= trank - trankx && idim < trank, "");
                std::get<idim + trankx - trank>(dimsx) = std::get<idim>(dims);
                dims_t<idim + 1, trank>::set(dims, dimsx);
            }
        };

        template <size_t trank>
        struct dims_t<trank, trank>
        {
            template <size_t trankx>
            static void set(const tensor_dims_t<trank>&, tensor_dims_t<trankx>&)
            {
            }
        };

        template <size_t trank, size_t trankx>
        void get_dims0(const tensor_dims_t<trank>& dims, tensor_dims_t<trankx>& dimsx)
        {
            dims_t<trank - trankx, trank>::set(dims, dimsx);
        }
    }

    ///
    /// \brief index a multi-dimensional tensor.
    ///
    template <size_t trank, typename... tindices>
    tensor_size_t index(const tensor_dims_t<trank>& dims, const tindices... indices)
    {
        static_assert(trank >= 1, "invalid number of tensor dimensions");
        static_assert(sizeof...(indices) == trank, "invalid number of tensor indices");
        return detail::get_index<0>(dims, indices...);
    }

    ///
    /// \brief index a multi-dimensional tensor (assuming the last dimensions that are ignored are zero).
    ///
    template <size_t trank, typename... tindices>
    tensor_size_t index0(const tensor_dims_t<trank>& dims, const tindices... indices)
    {
        static_assert(trank >= 1, "invalid number of tensor dimensions");
        static_assert(sizeof...(indices) <= trank, "invalid number of tensor indices");
        return detail::get_index0<0>(dims, indices...);
    }
    ///
    /// \brief gather the missing dimensions in a multi-dimensional tensor
    ///     (assuming the last dimensions that are ignored are zero).
    ///
    template <size_t trank, typename... tindices, size_t trankx = trank - sizeof...(tindices)>
    tensor_dims_t<trankx> dims0(const tensor_dims_t<trank>& dims, const tindices...)
    {
        static_assert(trank >= 1, "invalid number of tensor dimensions");
        static_assert(sizeof...(tindices) < trank, "invalid number of tensor indices");
        tensor_dims_t<trankx> dimsx;
        dimsx.fill(0);
        detail::get_dims0(dims, dimsx);
        return dimsx;
    }

    ///
    /// \brief size of a multi-dimensional tensor (#elements).
    ///
    template <size_t trank>
    tensor_size_t size(const tensor_dims_t<trank>& dims)
    {
        static_assert(trank >= 1, "invalid number of tensor dimensions");
        return detail::product<0>(dims);
    }

    ///
    /// \brief compare two tensor by dimension.
    ///
    template <size_t trank>
    bool operator==(const tensor_dims_t<trank>& dims1, const tensor_dims_t<trank>& dims2)
    {
        return std::operator==(dims1, dims2);
    }

    template <size_t trank>
    bool operator!=(const tensor_dims_t<trank>& dims1, const tensor_dims_t<trank>& dims2)
    {
        return std::operator!=(dims1, dims2);
    }

    ///
    /// \brief stream tensor dimensions.
    ///
    template <size_t trank>
    std::ostream& operator<<(std::ostream& os, const tensor_dims_t<trank>& dims)
    {
        for (size_t d = 0; d < dims.size(); ++ d)
        {
            os << dims[d] << (d + 1 == dims.size() ? "" : "x");
        }
        return os;
    }
}
