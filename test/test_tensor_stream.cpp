#include <fstream>
#include <nano/tensor/stream.h>
#include <sstream>
#include <utest/utest.h>

using namespace nano;

template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
static auto tensor2str(const tensor_t<tstorage, tscalar, trank>& tensor)
{
    std::ostringstream stream;
    UTEST_REQUIRE_NOTHROW(nano::write(stream, tensor));
    UTEST_REQUIRE(stream);
    return stream.str();
}

template <typename tscalar>
static auto make_tensor()
{
    tensor_mem_t<tscalar, 3> tensor(5, 3, 1);
    tensor.random(std::numeric_limits<tscalar>::lowest(), std::numeric_limits<tscalar>::max());
    return tensor;
}

template <typename tscalar>
static auto check_read_tensor(const std::string& str)
{
    tensor_mem_t<tscalar, 3> tensor;
    tensor.random();
    {
        std::istringstream stream(str);
        UTEST_REQUIRE(nano::read(stream, tensor));
        UTEST_REQUIRE(stream);
        UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());
    }
    return tensor;
}

UTEST_BEGIN_MODULE(test_tensor_stream)

UTEST_CASE(read_write_int32)
{
    const auto tensor = make_tensor<int32_t>();
    const auto str    = tensor2str(tensor);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(4 + 4 + 3 * 4 + 4 + 8 + tensor.size() * 4));

    const auto read_tensor = check_read_tensor<int32_t>(str);
    UTEST_CHECK_EQUAL(tensor, read_tensor);
}

UTEST_CASE(read_write_uint64)
{
    const auto tensor = make_tensor<uint64_t>();
    const auto str    = tensor2str(tensor);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(4 + 4 + 3 * 4 + 4 + 8 + tensor.size() * 8));

    const auto read_tensor = check_read_tensor<uint64_t>(str);
    UTEST_CHECK_EQUAL(tensor, read_tensor);
}

UTEST_CASE(read_write_float)
{
    const auto tensor = make_tensor<float>();
    const auto str    = tensor2str(tensor);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(4 + 4 + 3 * 4 + 4 + 8 + tensor.size() * 4));

    const auto read_tensor = check_read_tensor<float>(str);
    UTEST_CHECK_CLOSE(tensor, read_tensor, std::numeric_limits<float>::epsilon());
}

UTEST_CASE(read_write_double)
{
    const auto tensor = make_tensor<double>();
    const auto str    = tensor2str(tensor);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(4 + 4 + 3 * 4 + 4 + 8 + tensor.size() * 8));

    const auto read_tensor = check_read_tensor<double>(str);
    UTEST_CHECK_CLOSE(tensor, read_tensor, std::numeric_limits<double>::epsilon());
}

UTEST_CASE(write_fail)
{
    const auto tensor = make_tensor<int32_t>();

    std::ofstream stream;
    UTEST_CHECK(!nano::write(stream, tensor));
}

UTEST_CASE(read_fail_version)
{
    auto       read_tensor = make_tensor<int32_t>();
    const auto tensor      = make_tensor<int32_t>();

    auto str                                                      = tensor2str(tensor);
    reinterpret_cast<uint32_t*>(const_cast<char*>(str.data()))[0] = detail::hash_version() + 1U; // NOLINT

    std::istringstream stream(str);
    UTEST_CHECK(!nano::read(stream, read_tensor));
}

UTEST_CASE(read_fail_rank)
{
    auto       read_tensor = make_tensor<int32_t>();
    const auto tensor      = make_tensor<int32_t>();

    auto str                                                      = tensor2str(tensor);
    reinterpret_cast<uint32_t*>(const_cast<char*>(str.data()))[1] = uint32_t(1); // NOLINT

    std::istringstream stream(str);
    UTEST_CHECK(!nano::read(stream, read_tensor));
}

UTEST_CASE(read_fail_hash)
{
    auto       read_tensor = make_tensor<int32_t>();
    const auto tensor      = make_tensor<int32_t>();

    auto str                                                      = tensor2str(tensor);
    reinterpret_cast<uint32_t*>(const_cast<char*>(str.data()))[6] = uint32_t(13);     // NOLINT
    reinterpret_cast<uint32_t*>(const_cast<char*>(str.data()))[7] = uint32_t(124442); // NOLINT

    std::istringstream stream(str);
    UTEST_CHECK(!nano::read(stream, read_tensor));
}

UTEST_CASE(read_fail_out_of_range1)
{
    auto       read_tensor = make_tensor<int32_t>();
    const auto tensor      = make_tensor<int32_t>();

    auto str                                                     = tensor2str(tensor);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[2] = static_cast<int32_t>(tensor.size<0>() - 1); // NOLINT
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[3] = static_cast<int32_t>(tensor.size<1>() + 1); // NOLINT
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[4] = static_cast<int32_t>(tensor.size<2>() + 0); // NOLINT

    std::istringstream stream(str);
    UTEST_CHECK(!nano::read(stream, read_tensor));
}

UTEST_CASE(read_fail_out_of_range2)
{
    auto       read_tensor = make_tensor<int32_t>();
    const auto tensor      = make_tensor<int32_t>();

    auto str                                                     = tensor2str(tensor);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[2] = static_cast<int32_t>(tensor.size<0>() + 1); // NOLINT
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[3] = static_cast<int32_t>(tensor.size<1>() + 1); // NOLINT
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[4] = static_cast<int32_t>(tensor.size<2>() + 1); // NOLINT

    std::istringstream stream(str);
    UTEST_CHECK(!nano::read(stream, read_tensor));
}

UTEST_CASE(read_fail_sizeof_scalar)
{
    auto       read_tensor = make_tensor<int32_t>();
    const auto tensor      = make_tensor<int32_t>();

    auto str                                                      = tensor2str(tensor);
    reinterpret_cast<uint32_t*>(const_cast<char*>(str.data()))[5] = // NOLINT
        static_cast<uint32_t>(sizeof(int32_t) + 1);

    std::istringstream stream(str);
    UTEST_CHECK(!nano::read(stream, read_tensor));
}

UTEST_END_MODULE()
