#pragma once

#include <vector>

namespace nano
{
    // sizes and indices
    using size_t = std::size_t;
    using sizes_t = std::vector<size_t>;
    using indices_t = std::vector<size_t>;

    // default scalar
    using scalar_t = double;
    using scalars_t = std::vector<scalar_t>;

    template <typename... tscalar>
    scalars_t make_scalars(tscalar... scalars)
    {
        return scalars_t{static_cast<scalar_t>(scalars)...};
    }
}
