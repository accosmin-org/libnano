#include <nano/core/stats.h>
#include <nano/mlearn/stats.h>

using namespace nano;
using namespace nano::ml;

namespace
{
auto percentile(const tensor1d_map_t& values, const double percentage)
{
    return ::nano::percentile(std::begin(values), std::end(values), percentage);
}
} // namespace

void nano::ml::store_stats(const tensor1d_map_t& values, const tensor1d_map_t& stats)
{
    stats(0)  = values.mean();
    stats(1)  = values.stdev();
    stats(2)  = static_cast<scalar_t>(values.size());
    stats(3)  = ::percentile(values, 1.0);
    stats(4)  = ::percentile(values, 5.0);
    stats(5)  = ::percentile(values, 10.0);
    stats(6)  = ::percentile(values, 20.0);
    stats(7)  = ::percentile(values, 50.0);
    stats(8)  = ::percentile(values, 80.0);
    stats(9)  = ::percentile(values, 90.0);
    stats(10) = ::percentile(values, 95.0);
    stats(11) = ::percentile(values, 99.0);
}

stats_t nano::ml::load_stats(const tensor1d_cmap_t& stats)
{
    assert(stats.size() == 12);

    return {
        stats(0), stats(1), stats(2), stats(3), stats(4),  stats(5),
        stats(6), stats(7), stats(8), stats(9), stats(10), stats(11),
    };
}
