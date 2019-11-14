#pragma once

#include <istream>
#include <ostream>
#include <functional>
#include <nano/tensor/tensor.h>

namespace nano
{
    namespace detail
    {
        template <typename tscalar>
        std::ostream& write(std::ostream& os, const tscalar scalar)
        {
            return os.write(reinterpret_cast<const char*>(&scalar), sizeof(tscalar)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        }

        template <typename tscalar, typename tcount>
        std::ostream& write(std::ostream& os, const tscalar* data, const tcount count)
        {
            return os.write(reinterpret_cast<const char*>(data), sizeof(tscalar) * count); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        }

        template <typename twscalar, typename tscalar, typename tcount>
        std::ostream& write_cast(std::ostream& os, const tscalar* data, const tcount count)
        {
            for (tcount i = 0; i < count; ++ i)
            {
                write(os, static_cast<twscalar>(data[i]));
            }
            return os;
        }

        template <typename tscalar>
        std::istream& read(std::istream& is, tscalar& scalar)
        {
            return is.read(reinterpret_cast<char*>(&scalar), sizeof(tscalar)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        }

        template <typename tscalar, typename tcount>
        std::istream& read(std::istream& is, tscalar* data, const tcount count)
        {
            return is.read(reinterpret_cast<char*>(data), sizeof(tscalar) * count); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        }

        template <typename trscalar, typename tscalar, typename tcount>
        std::istream& read_cast(std::istream& is, tscalar* data, const tcount count)
        {
            for (tcount i = 0; i < count; ++ i)
            {
                trscalar value{};
                read(is, value);
                data[i] = static_cast<tscalar>(value);
            }
            return is;
        }

        inline uint64_t hash_combine(const uint64_t h1, const uint64_t h2)
        {
            return h1 ^ (h2 << 1U);
        }

        template <typename tscalar>
        uint64_t hash(const tscalar* data, const tensor_size_t size)
        {
            const auto hasher = std::hash<tscalar>{};

            uint64_t hash = 0;
            for (tensor_size_t i = 0 ; i < size; ++ i)
            {
                hash = hash_combine(hash, hasher(data[i]));
            }
            return hash;
        }
    }

    ///
    /// \brief write unformatted the given tensor.
    ///
    template <typename tstorage, size_t trank>
    std::ostream& write(std::ostream& os, const tensor_t<tstorage, trank>& tensor)
    {
        using tscalar = typename tensor_t<tstorage, trank>::tscalar;

        if (!detail::write(os, static_cast<uint32_t>(trank)) ||                 // rank
            !detail::write_cast<int32_t>(os, tensor.dims().data(), trank) ||    // dimensions
            !detail::write(os, static_cast<uint32_t>(sizeof(tscalar))) ||       // sizeof(scalar)
            !detail::write(os, detail::hash(tensor.data(), tensor.size())) ||   // hash(content)
            !detail::write(os, tensor.data(), tensor.size()))                   // content
        {
            os.setstate(std::ios_base::failbit);
        }
        return os;
    }

    ///
    /// \brief read unformatted the given tensor.
    ///
    template <typename tstorage, size_t trank>
    std::istream& read(std::istream& is, tensor_t<tstorage, trank>& tensor)
    {
        using tscalar = typename tensor_t<tstorage, trank>::tscalar;
        typename tensor_t<tstorage, trank>::tdims dims;
        uint32_t irank = 0, iscalar = 0;
        uint64_t ihash = 0;

        if (!detail::read(is, irank) ||                                         // rank
            !detail::read_cast<int32_t>(is, dims.data(), trank) ||              // dimensions
            !detail::read(is, iscalar) ||                                       // sizeof(scalar)
            !detail::read(is, ihash) ||                                         // hash(content)
            static_cast<size_t>(irank) != trank ||
            static_cast<size_t>(iscalar) != sizeof(tscalar))
        {
            is.setstate(std::ios_base::failbit);
            return is;
        }

        tensor.resize(dims);
        if (!detail::read(is, tensor.data(), tensor.size()) ||                  // content
            ihash != detail::hash(tensor.data(), tensor.size()))
        {
            is.setstate(std::ios_base::failbit);
        }
        return is;
    }
}
