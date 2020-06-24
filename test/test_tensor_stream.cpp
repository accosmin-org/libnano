#include <fstream>
#include <sstream>
#include <utest/utest.h>
#include <nano/tensor/stream.h>

using namespace nano;

template <typename tstorage, size_t trank>
auto tensor2str(const tensor_t<tstorage, trank>& tensor)
{
    std::ostringstream stream;
    UTEST_REQUIRE_NOTHROW(nano::write(stream, tensor));
    UTEST_REQUIRE(stream);
    return stream.str();
}

auto make_tensor()
{
    tensor_mem_t<int32_t, 3> tensor(5, 3, 1);
    tensor.random(-100, +100);
    return tensor;
}

UTEST_BEGIN_MODULE(test_tensor_stream)

UTEST_CASE(read_write)
{
    const auto tensor = make_tensor();

    const auto str = tensor2str(tensor);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(4 + 4 + 3 * 4 + 4 + 8 + 5 * 3 * 1 * 4));

    tensor_mem_t<int32_t, 3> read_tensor;
    read_tensor.random();
    {
        std::istringstream stream(str);
        UTEST_REQUIRE(nano::read(stream, read_tensor));
        UTEST_REQUIRE(stream);
        UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());
    }

    UTEST_CHECK_EQUAL(tensor.dims(), read_tensor.dims());
    UTEST_CHECK_EQUAL(tensor.vector(), read_tensor.vector());
}

UTEST_CASE(write_fail)
{
    const auto tensor = make_tensor();

    std::ofstream stream;
    UTEST_CHECK(!nano::write(stream, tensor));
}

UTEST_CASE(read_fail_version)
{
    auto read_tensor = make_tensor();
    const auto tensor = make_tensor();

    auto str = tensor2str(tensor);
    reinterpret_cast<uint32_t*>(const_cast<char*>(str.data()))[0] = detail::tensor_version() + 1U; // NOLINT

    std::istringstream stream(str);
    UTEST_CHECK(!nano::read(stream, read_tensor));
}

UTEST_CASE(read_fail_rank)
{
    auto read_tensor = make_tensor();
    const auto tensor = make_tensor();

    auto str = tensor2str(tensor);
    reinterpret_cast<uint32_t*>(const_cast<char*>(str.data()))[1] = uint32_t(1); // NOLINT

    std::istringstream stream(str);
    UTEST_CHECK(!nano::read(stream, read_tensor));
}

UTEST_CASE(read_fail_hash)
{
    auto read_tensor = make_tensor();
    const auto tensor = make_tensor();

    auto str = tensor2str(tensor);
    reinterpret_cast<uint32_t*>(const_cast<char*>(str.data()))[6] = uint32_t(13); // NOLINT
    reinterpret_cast<uint32_t*>(const_cast<char*>(str.data()))[7] = uint32_t(124442); // NOLINT

    std::istringstream stream(str);
    UTEST_CHECK(!nano::read(stream, read_tensor));
}

UTEST_CASE(read_fail_out_of_range1)
{
    auto read_tensor = make_tensor();
    const auto tensor = make_tensor();

    auto str = tensor2str(tensor);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[2] = static_cast<int32_t>(tensor.size<0>() - 1); // NOLINT
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[3] = static_cast<int32_t>(tensor.size<1>() + 1); // NOLINT
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[4] = static_cast<int32_t>(tensor.size<2>() + 0); // NOLINT

    std::istringstream stream(str);
    UTEST_CHECK(!nano::read(stream, read_tensor));
}

UTEST_CASE(read_fail_out_of_range2)
{
    auto read_tensor = make_tensor();
    const auto tensor = make_tensor();

    auto str = tensor2str(tensor);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[2] = static_cast<int32_t>(tensor.size<0>() + 1); // NOLINT
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[3] = static_cast<int32_t>(tensor.size<1>() + 1); // NOLINT
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[4] = static_cast<int32_t>(tensor.size<2>() + 1); // NOLINT

    std::istringstream stream(str);
    UTEST_CHECK(!nano::read(stream, read_tensor));
}

UTEST_CASE(read_fail_sizeof_scalar)
{
    auto read_tensor = make_tensor();
    const auto tensor = make_tensor();

    auto str = tensor2str(tensor);
    reinterpret_cast<uint32_t*>(const_cast<char*>(str.data()))[5] = static_cast<uint32_t>(sizeof(int32_t) + 1); // NOLINT

    std::istringstream stream(str);
    UTEST_CHECK(!nano::read(stream, read_tensor));
}

UTEST_END_MODULE()
