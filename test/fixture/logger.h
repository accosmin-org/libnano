#include <nano/logger.h>

namespace
{
template <class... targs>
void log_info(const targs&... args)
{
    make_stdout_logger().log(log_type::info, args...);
}
} // namespace
