#include <nano/wlearner/mhash.h>
#include <set>

using namespace nano;

mhashes_t nano::make_mhashes(const mclass_cmap_t& fvalues)
{
    std::set<uint64_t> fhashes;
    for (tensor_size_t i = 0, size = fvalues.size<0>(); i < size; ++i)
    {
        const auto values = fvalues.array(i);
        if (values(0) < 0)
        {
            continue;
        }

        const auto hash = ::mhash(values);
        fhashes.insert(hash);
    }

    auto mhashes = mhashes_t{static_cast<tensor_size_t>(fhashes.size())};
    std::copy(fhashes.begin(), fhashes.end(), mhashes.begin());
    return mhashes;
}
