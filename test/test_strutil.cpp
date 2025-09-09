#include <iomanip>
#include <nano/core/strutil.h>
#include <utest/utest.h>

namespace nano
{
enum class enum_type : uint8_t
{
    type1,
    type2,
    type3
};

template <>
enum_map_t<nano::enum_type> enum_string<nano::enum_type>() // cppcheck-suppress unusedFunction
{
    return {
        {enum_type::type1, "type1"},
        {enum_type::type3, "type3"}
    };
}
} // namespace nano

UTEST_BEGIN_MODULE()

UTEST_CASE(scat)
{
    UTEST_CHECK_EQUAL(nano::scat(1), "1");
    UTEST_CHECK_EQUAL(nano::scat(124545), "124545");
    UTEST_CHECK_EQUAL(nano::scat(nano::string_t("str"), "x", 'a', 42, nano::string_t("end")), "strxa42end");
    UTEST_CHECK_EQUAL(nano::scat("str", nano::string_t("x"), 'a', 42, nano::string_t("end")), "strxa42end");
    UTEST_CHECK_EQUAL(nano::scat(nano::enum_type::type1, "str", nano::enum_type::type3, 42), "type1strtype342");
    UTEST_CHECK_EQUAL(nano::scat("str", std::setprecision(0), std::fixed, 1.42, nano::string_t("F")), "str1F");
    UTEST_CHECK_EQUAL(nano::scat("str", std::setprecision(1), std::fixed, 1.42, nano::string_t("F")), "str1.4F");

    const auto values = std::vector<int>{1, 2, 3};
    UTEST_CHECK_EQUAL(nano::scat("[", values, "]"), "[1,2,3]");
}

UTEST_CASE(from_string)
{
    UTEST_CHECK_EQUAL(nano::from_string<short>("1"), 1);
    UTEST_CHECK_EQUAL(nano::from_string<float>("0.2f"), 0.2F);
    UTEST_CHECK_EQUAL(nano::from_string<long int>("124545"), 124545);
    UTEST_CHECK_EQUAL(nano::from_string<unsigned long>("42"), 42U);

    UTEST_CHECK_THROW(nano::from_string<int>("x"), std::invalid_argument);
    UTEST_CHECK_THROW(nano::from_string<short>("1000000000"), std::out_of_range);
}

UTEST_CASE(enum_string)
{
    UTEST_CHECK_EQUAL(nano::scat(nano::enum_type::type1), "type1");
    UTEST_CHECK_THROW(nano::scat(nano::enum_type::type2), std::invalid_argument);
    UTEST_CHECK_EQUAL(nano::scat(nano::enum_type::type3), "type3");

    UTEST_CHECK(nano::from_string<nano::enum_type>("type1") == nano::enum_type::type1);
    UTEST_CHECK(nano::from_string<nano::enum_type>("type3") == nano::enum_type::type3);
    UTEST_CHECK(nano::from_string<nano::enum_type>("type3[") == nano::enum_type::type3);

    UTEST_CHECK_THROW(nano::from_string<nano::enum_type>("????"), std::invalid_argument);
    UTEST_CHECK_THROW(nano::from_string<nano::enum_type>("type"), std::invalid_argument);
    UTEST_CHECK_THROW(nano::from_string<nano::enum_type>("type2"), std::invalid_argument);
}

UTEST_CASE(enum_values)
{
    const auto enums13 = std::vector<nano::enum_type>{nano::enum_type::type1, nano::enum_type::type3};
    UTEST_CHECK_EQUAL(nano::enum_values<nano::enum_type>(), enums13);

    const auto enums3 = std::vector<nano::enum_type>{nano::enum_type::type3};
    UTEST_CHECK_EQUAL(nano::enum_values<nano::enum_type>(std::regex(".+3")), enums3);
}

UTEST_CASE(resize)
{
    UTEST_CHECK_EQUAL(nano::align("text", 10, nano::alignment::left, '='), "text======");
    UTEST_CHECK_EQUAL(nano::align("text", 10, nano::alignment::right, '='), "======text");
    UTEST_CHECK_EQUAL(nano::align("text", 10, nano::alignment::left, '='), "text======");
    UTEST_CHECK_EQUAL(nano::align("text", 10, nano::alignment::center, '='), "===text===");
}

UTEST_CASE(ends_with)
{
    UTEST_CHECK(nano::ends_with("ToKeN", ""));
    UTEST_CHECK(nano::ends_with("ToKeN", "N"));
    UTEST_CHECK(nano::ends_with("ToKeN", "eN"));
    UTEST_CHECK(nano::ends_with("ToKeN", "KeN"));
    UTEST_CHECK(nano::ends_with("ToKeN", "oKeN"));
    UTEST_CHECK(nano::ends_with("ToKeN", "ToKeN"));

    UTEST_CHECK(!nano::ends_with("ToKeN", "n"));
    UTEST_CHECK(!nano::ends_with("ToKeN", "en"));
    UTEST_CHECK(!nano::ends_with("ToKeN", "ken"));
    UTEST_CHECK(!nano::ends_with("ToKeN", "oken"));
    UTEST_CHECK(!nano::ends_with("ToKeN", "Token"));
}

UTEST_CASE(starts_with)
{
    UTEST_CHECK(nano::starts_with("ToKeN", ""));
    UTEST_CHECK(nano::starts_with("ToKeN", "T"));
    UTEST_CHECK(nano::starts_with("ToKeN", "To"));
    UTEST_CHECK(nano::starts_with("ToKeN", "ToK"));
    UTEST_CHECK(nano::starts_with("ToKeN", "ToKe"));
    UTEST_CHECK(nano::starts_with("ToKeN", "ToKeN"));

    UTEST_CHECK(!nano::starts_with("ToKeN", "t"));
    UTEST_CHECK(!nano::starts_with("ToKeN", "to"));
    UTEST_CHECK(!nano::starts_with("ToKeN", "tok"));
    UTEST_CHECK(!nano::starts_with("ToKeN", "toke"));
    UTEST_CHECK(!nano::starts_with("ToKeN", "Token"));
}

UTEST_END_MODULE()
