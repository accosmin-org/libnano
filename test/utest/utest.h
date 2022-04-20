#pragma once

#include <cmath>
#include <atomic>
#include <string>
#include <vector>
#include <utility>
#include <iomanip>
#include <iostream>
#include <nano/tensor/eigen.h>
#include <nano/tensor/tensor.h>

#define UTEST_STRINGIFY_(x) #x
#define UTEST_STRINGIFY(x) UTEST_STRINGIFY_(x)

#define ERROR_COLOR "\033[35m"
#define RESET_COLOR "\033[0m"

static std::string utest_test_name;
static std::string utest_case_name;
static std::string utest_module_name;

static std::size_t utest_n_cases = 0;
static std::atomic<std::size_t> utest_n_checks = {0};
static std::atomic<std::size_t> utest_n_failures = {0};

struct utest_location_t
{
};

inline std::ostream& operator<<(std::ostream& stream, const utest_location_t&)
{
    stream << "[" << utest_module_name << "/" << utest_case_name;
    if (!utest_test_name.empty())
    {
        stream << "/" << utest_test_name;
    }
    return stream << "]";
}

struct utest_test_name_t
{
    explicit utest_test_name_t(std::string test_name)
    {
        utest_test_name = std::move(test_name);
    }

    ~utest_test_name_t()
    {
        utest_test_name.clear();
    }
};

enum class exception_status
{
    none,
    expected,
    unexpected
};

template <typename texception, typename toperator>
static exception_status check_throw(const toperator& op)
{
    try
    {
        op();
        return exception_status::none;
    }
    catch (texception& e)
    {
        return exception_status::expected;
    }
    catch (std::exception& e)
    {
        return exception_status::unexpected;
    }
    catch (...)
    {
        return exception_status::unexpected;
    }
}

template <class T>
struct is_pair : std::false_type
{
};

template <class T1, class T2>
struct is_pair<std::pair<T1, T2>> : std::true_type
{
};

template <class T>
inline constexpr bool is_pair_v = is_pair<T>::value;

template <typename tvalue>
static std::ostream& operator<<(std::ostream& os, const std::vector<tvalue>& values)
{
    os << "{";
    for (const auto& value : values)
    {
        if constexpr(std::is_arithmetic_v<tvalue> || std::is_same_v<tvalue, std::string>)
        {
            os << "{" << value << "}";
        }
        else if constexpr(is_pair_v<tvalue>)
        {
            os << "{" << value.first << "," << value.second << "}";
        }
    }
    return os << "}";
}

#define UTEST_BEGIN_MODULE(name) \
int main(int, char* []) /*NOLINT(hicpp-function-size,readability-function-size)*/ \
{ \
try \
{ \
    utest_module_name = #name;

#define UTEST_CASE(name) \
    ++ utest_n_cases; \
    utest_case_name = #name; \
    std::cout << "running test case " << utest_location_t{} << " ..." << std::endl;

#define UTEST_END_MODULE() \
    if (utest_n_failures > 0) \
    { \
        std::cout << " failed with " << utest_n_failures << " errors in " << utest_n_checks \
              << " check" << (utest_n_checks > 0 ? "s" : "") << "!" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
    else \
    { \
        std::cout << "  no errors detected in " << utest_n_checks \
              << " check" << (utest_n_checks > 0 ? "s" : "") << "." << std::endl; \
        exit(EXIT_SUCCESS); \
    } \
} \
catch (std::exception& e) \
{ \
    std::cout << " failed with uncaught exception <" << e.what() << ">!" << std::endl; \
    std::cout << " failed with " << utest_n_failures << " errors in " << utest_n_checks \
          << " check" << (utest_n_checks > 0 ? "s" : "") << "!" << std::endl; \
    exit(EXIT_FAILURE); \
} \
catch (...) \
{ \
    std::cout << " failed with uncaught unknown exception!" << std::endl; \
    std::cout << " failed with " << utest_n_failures << " errors in " << utest_n_checks \
          << " check" << (utest_n_checks > 0 ? "s" : "") << "!" << std::endl; \
    exit(EXIT_FAILURE); \
} \
}

#define UTEST_HANDLE_CRITICAL(critical) \
    if (critical) \
    { \
        exit(EXIT_FAILURE); \
    }
#define UTEST_HANDLE_FAILURE() \
    ++ utest_n_failures; \
    std::cout << ERROR_COLOR << __FILE__ << ":" << __LINE__ \
        << std::fixed << std::setprecision(12) << ": " << utest_location_t{} << ": "

#define UTEST_EVALUATE(check, critical) \
    ++ utest_n_checks; \
    if (!(check)) \
    { \
        UTEST_HANDLE_FAILURE() \
            << "check {" << UTEST_STRINGIFY(check) << "} failed!" << RESET_COLOR << std::endl; \
        UTEST_HANDLE_CRITICAL(critical) \
    }
#define UTEST_CHECK(check) \
    UTEST_EVALUATE(check, false);
#define UTEST_REQUIRE(check) \
    UTEST_EVALUATE(check, true);

#define UTEST_THROW(call, exception, critical) \
    ++ utest_n_checks; \
    switch (check_throw<exception>([&] () { (void)(call); })) \
    { \
    case exception_status::none: \
        UTEST_HANDLE_FAILURE() \
            << "call {" << UTEST_STRINGIFY(call) << "} does not throw!" << RESET_COLOR << std::endl; \
        UTEST_HANDLE_CRITICAL(critical) \
    case exception_status::expected: \
        break; \
    case exception_status::unexpected: \
        UTEST_HANDLE_FAILURE() \
            << "call {" << UTEST_STRINGIFY(call) << "} does not throw {" \
            << UTEST_STRINGIFY(exception) << "}!" << RESET_COLOR << std::endl; \
        UTEST_HANDLE_CRITICAL(critical) \
    }
#define UTEST_CHECK_THROW(call, exception) \
    UTEST_THROW(call, exception, false)
#define UTEST_REQUIRE_THROW(call, exception) \
    UTEST_THROW(call, exception, true)

#define UTEST_NOTHROW(call, critical) \
    ++ utest_n_checks; \
    switch (check_throw<std::exception>([&] () { (void)(call); })) \
    { \
    case exception_status::none: \
        break; \
    case exception_status::expected: \
    case exception_status::unexpected: \
        UTEST_HANDLE_FAILURE() \
            << "call {" << UTEST_STRINGIFY(call) << "} throws!" << RESET_COLOR << std::endl; \
        UTEST_HANDLE_CRITICAL(critical) \
    }
#define UTEST_CHECK_NOTHROW(call) \
    UTEST_NOTHROW(call, false)
#define UTEST_REQUIRE_NOTHROW(call) \
    UTEST_NOTHROW(call, true)

#define UTEST_EVALUATE_BINARY_OP(left, right, op, critical) \
{ \
    ++ utest_n_checks; \
    const auto res_left = (left); /* NOLINT */ \
    const auto res_right = (right); /* NOLINT */ \
    if (!(res_left op res_right)) \
    { \
        UTEST_HANDLE_FAILURE() \
            << "check {" << UTEST_STRINGIFY(left op right) \
            << "} failed {" << res_left << " " << UTEST_STRINGIFY(op) \
            << " " << res_right << "}!" << RESET_COLOR << std::endl; \
        UTEST_HANDLE_CRITICAL(critical) \
    } \
}

#define UTEST_EVALUATE_EQUAL(left, right, critical) \
    UTEST_EVALUATE_BINARY_OP((left), (right), ==, critical);

#define UTEST_CHECK_EQUAL(left, right) \
    UTEST_EVALUATE_EQUAL(left, right, false);
#define UTEST_REQUIRE_EQUAL(left, right) \
    UTEST_EVALUATE_EQUAL(left, right, true);

#define UTEST_EVALUATE_NOT_EQUAL(left, right, critical) \
    UTEST_EVALUATE_BINARY_OP(left, right, !=, critical)
#define UTEST_CHECK_NOT_EQUAL(left, right) \
    UTEST_EVALUATE_NOT_EQUAL(left, right, false)
#define UTEST_REQUIRE_NOT_EQUAL(left, right) \
    UTEST_EVALUATE_NOT_EQUAL(left, right, true)

#define UTEST_EVALUATE_LESS(left, right, critical) \
    UTEST_EVALUATE_BINARY_OP(left, right, <, critical)
#define UTEST_CHECK_LESS(left, right) \
    UTEST_EVALUATE_LESS(left, right, false)
#define UTEST_REQUIRE_LESS(left, right) \
    UTEST_EVALUATE_LESS(left, right, true)

#define UTEST_EVALUATE_LESS_EQUAL(left, right, critical) \
    UTEST_EVALUATE_BINARY_OP(left, right, <=, critical)
#define UTEST_CHECK_LESS_EQUAL(left, right) \
    UTEST_EVALUATE_LESS_EQUAL(left, right, false)
#define UTEST_REQUIRE_LESS_EQUAL(left, right) \
    UTEST_EVALUATE_LESS_EQUAL(left, right, true)

#define UTEST_EVALUATE_GREATER(left, right, critical) \
    UTEST_EVALUATE_BINARY_OP(left, right, >, critical);
#define UTEST_CHECK_GREATER(left, right) \
    UTEST_EVALUATE_GREATER(left, right, false);
#define UTEST_REQUIRE_GREATER(left, right) \
    UTEST_EVALUATE_GREATER(left, right, true);

#define UTEST_EVALUATE_GREATER_EQUAL(left, right, critical) \
    UTEST_EVALUATE_BINARY_OP(left, right, >=, critical);
#define UTEST_CHECK_GREATER_EQUAL(left, right) \
    UTEST_EVALUATE_GREATER_EQUAL(left, right, false);
#define UTEST_REQUIRE_GREATER_EQUAL(left, right) \
    UTEST_EVALUATE_GREATER_EQUAL(left, right, true);

#define UTEST_EVALUATE_CLOSE(left, right, epsilon, critical) \
    ++ utest_n_checks; \
    if (!::nano::close((left), (right), epsilon)) \
    { \
        UTEST_HANDLE_FAILURE() \
            << "check {" << UTEST_STRINGIFY(left <> right) \
            << "} failed {" << (left) << " <> " << (right) << "}!" << RESET_COLOR << std::endl; \
        UTEST_HANDLE_CRITICAL(critical); \
    }

#define UTEST_CHECK_CLOSE(left, right, epsilon) \
    UTEST_EVALUATE_CLOSE(left, right, epsilon, false);
#define UTEST_REQUIRE_CLOSE(left, right, epsilon) \
    UTEST_EVALUATE_CLOSE(left, right, epsilon, true);
