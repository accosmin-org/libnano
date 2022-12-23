#include <nano/dataset/hash.h>
#include <set>

using namespace nano;

static hashes_t make_hashes(const std::set<uint64_t>& fhashes)
{
    auto hashes = hashes_t{static_cast<tensor_size_t>(fhashes.size())};
    std::copy(fhashes.begin(), fhashes.end(), hashes.begin());
    return hashes;
}

hashes_t nano::make_hashes(const sclass_cmap_t& fvalues)
{
    std::set<uint64_t> fhashes;
    for (tensor_size_t i = 0, size = fvalues.size(); i < size; ++i)
    {
        const auto value = fvalues(i);
        if (value >= 0)
        {
            fhashes.insert(::nano::hash(value));
        }
    }

    return ::make_hashes(fhashes);
}

hashes_t nano::make_hashes(const mclass_cmap_t& fvalues)
{
    std::set<uint64_t> fhashes;
    for (tensor_size_t i = 0, size = fvalues.size<0>(); i < size; ++i)
    {
        const auto values = fvalues.array(i);
        if (values(0) >= 0)
        {
            fhashes.insert(::nano::hash(values));
        }
    }

    return ::make_hashes(fhashes);
}
