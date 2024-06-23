#pragma once

#include <nano/logger.h>
#include <stdexcept>

namespace nano
{
///
/// \brief wraps main function to catch and log all exceptions.
///
template <class tcallback>
int safe_main(const tcallback& callback, const int argc, const char* argv[])
{
    try
    {
        return callback(argc, argv);
    }
    catch (const std::exception& e)
    {
        make_stdout_logger().log(log_type::error, "caught exception (", e.what(), ")!\n");
        return EXIT_FAILURE;
    }
    catch (...)
    {
        make_stdout_logger().log(log_type::error, "caught unknown exception!\n");
        return EXIT_FAILURE;
    }
}
} // namespace nano
