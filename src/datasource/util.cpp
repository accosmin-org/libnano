#include <cstdlib>
#include <nano/datasource/util.h>

using namespace nano;

string_t nano::getenv(const char* env_var_name)
{
#ifdef _WIN32
    size_t requiredSize = 0U;
    getenv_s(&requiredSize, nullptr, 0, env_var_name);
    if (requiredSize == 0)
    {
        return string_t{};
    }

    std::vector<char> env_var_value(requiredSize);
    getenv_s(&requiredSize, env_var_value.data(), requiredSize, env_var_name);

    return string_t{env_var_value.data(), env_var_value.data() + requiredSize};
#else
    // getenv_s is not available on POSIX by default, use not thread-safe getenv!
    const char* env_var_value = std::getenv(env_var_name); // NOLINT(concurrency-mt-unsafe)
    return env_var_value == nullptr ? string_t{} : string_t{env_var_value};
#endif
}
