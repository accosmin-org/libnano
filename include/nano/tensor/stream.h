#pragma once

#include <functional>
#include <nano/stream.h>
#include <nano/tensor/tensor.h>

namespace nano
{
    namespace detail
    {
        inline constexpr uint32_t tensor_version()
        {
            return 0;
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
    std::ostream& write(std::ostream& stream, const tensor_t<tstorage, trank>& tensor)
    {
        using tscalar = typename tensor_t<tstorage, trank>::tscalar;

        if (!::nano::write(stream, detail::tensor_version()) ||                     // version
            !::nano::write(stream, static_cast<uint32_t>(trank)) ||                 // rank
            !::nano::write_cast<int32_t>(stream, tensor.dims().data(), trank) ||    // dimensions
            !::nano::write(stream, static_cast<uint32_t>(sizeof(tscalar))) ||       // sizeof(scalar)
            !::nano::write(stream, detail::hash(tensor.data(), tensor.size())) ||   // hash(content)
            !::nano::write(stream, tensor.data(), tensor.size()))                   // content
        {
            stream.setstate(std::ios_base::failbit);
        }
        return stream;
    }

    ///
    /// \brief read unformatted the given tensor.
    ///
    template <typename tstorage, size_t trank>
    std::istream& read(std::istream& stream, tensor_t<tstorage, trank>& tensor)
    {
        using tscalar = typename tensor_t<tstorage, trank>::tscalar;
        typename tensor_t<tstorage, trank>::tdims dims;
        uint32_t iversion = 0xFFFFFFFF, irank = 0, iscalar = 0;
        uint64_t ihash = 0;

        if (!::nano::read(stream, iversion) ||                                      // version
            !::nano::read(stream, irank) ||                                         // rank
            !::nano::read_cast<int32_t>(stream, dims.data(), trank) ||              // dimensions
            !::nano::read(stream, iscalar) ||                                       // sizeof(scalar)
            !::nano::read(stream, ihash) ||                                         // hash(content)
            iversion != detail::tensor_version() ||
            static_cast<size_t>(irank) != trank ||
            static_cast<size_t>(iscalar) != sizeof(tscalar))
        {
            stream.setstate(std::ios_base::failbit);
            return stream;
        }

        tensor.resize(dims);
        if (!::nano::read(stream, tensor.data(), tensor.size()) ||                  // content
            ihash != detail::hash(tensor.data(), tensor.size()))
        {
            stream.setstate(std::ios_base::failbit);
        }
        return stream;
    }
}
