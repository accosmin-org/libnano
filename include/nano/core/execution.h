#pragma once

#include <cstdint>

namespace nano
{
    ///
    /// \brief execution policy.
    ///
    enum class execution : int32_t
    {
        seq = 0,        ///< sequential: using only the thread
        par             ///< parallel: use all the available threads
    };
}
