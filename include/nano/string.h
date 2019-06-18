#pragma once

#include <vector>
#include <string>

namespace nano
{
    // strings
    using string_t = std::string;
    using strings_t = std::vector<string_t>;

    ///
    /// \brief text alignment options
    ///
    enum class alignment : int
    {
        left,
        center,
        right
    };
}
