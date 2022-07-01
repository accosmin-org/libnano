#pragma once

#include <cstdlib>

namespace nano
{
    inline char* getenv(const char* env_var)
    {
        return std::getenv(env_var); // NOLINT(concurrency-mt-unsafe)
    }
} // namespace nano
