#include <set>
#include <list>
#include <utest/utest.h>
#include <nano/string.h>

enum class enum_type
{
    type1,
    type2,
    type3
};

namespace nano
{
    template <>
    enum_map_t<enum_type> enum_string<enum_type>()
    {
        return
        {
            { enum_type::type1,     "type1" },
//                { enum_type::type2,     "type2" },
            { enum_type::type3,     "type3" }
        };
    }
}

std::ostream& operator<<(std::ostream& os, const std::vector<enum_type>& enums)
{
    for (const auto& e : enums)
    {
        os << nano::to_string(e) << " ";
    }
    return os;
}

UTEST_BEGIN_MODULE(test_string)

UTEST_CASE(to_string)
{
    UTEST_CHECK_EQUAL(nano::to_string(1), "1");
    UTEST_CHECK_EQUAL(nano::to_string(124545), "124545");
}

UTEST_CASE(from_string)
{
    UTEST_CHECK_EQUAL(nano::from_string<short>("1"), 1);
    UTEST_CHECK_EQUAL(nano::from_string<float>("0.2f"), 0.2f);
    UTEST_CHECK_EQUAL(nano::from_string<long int>("124545"), 124545);
    UTEST_CHECK_EQUAL(nano::from_string<unsigned long>("42"), 42u);
}

UTEST_CASE(enum_string)
{
    UTEST_CHECK_EQUAL(nano::to_string(enum_type::type1), "type1");
    UTEST_CHECK_THROW(nano::to_string(enum_type::type2), std::invalid_argument);
    UTEST_CHECK_EQUAL(nano::to_string(enum_type::type3), "type3");

    UTEST_CHECK(nano::from_string<enum_type>("type1") == enum_type::type1);
    UTEST_CHECK(nano::from_string<enum_type>("type3") == enum_type::type3);
    UTEST_CHECK(nano::from_string<enum_type>("type3[") == enum_type::type3);

    UTEST_CHECK_THROW(nano::from_string<enum_type>("????"), std::invalid_argument);
    UTEST_CHECK_THROW(nano::from_string<enum_type>("type"), std::invalid_argument);
    UTEST_CHECK_THROW(nano::from_string<enum_type>("type2"), std::invalid_argument);
}

UTEST_CASE(enum_values)
{
    const auto enums13 = std::vector<enum_type>{enum_type::type1, enum_type::type3};
    UTEST_CHECK_EQUAL(nano::enum_values<enum_type>(), enums13);

    const auto enums3 = std::vector<enum_type>{enum_type::type3};
    UTEST_CHECK_EQUAL(nano::enum_values<enum_type>(std::regex(".+3")), enums3);
}

UTEST_CASE(join)
{
    UTEST_CHECK_EQUAL(nano::join(std::vector<int>({ 1, 2, 3 }), "-", nullptr, nullptr),      "1-2-3");
    UTEST_CHECK_EQUAL(nano::join(std::list<int>({ 1, 2, 3 }), "=", nullptr, nullptr),        "1=2=3");
    UTEST_CHECK_EQUAL(nano::join(std::set<int>({ 1, 2, 3 }), ",", nullptr, nullptr),         "1,2,3");

    UTEST_CHECK_EQUAL(nano::join(std::vector<int>({ 1, 2, 3 }), "-", "{", "}"),              "{1-2-3}");
    UTEST_CHECK_EQUAL(nano::join(std::list<int>({ 1, 2, 3 }), "=", "XXX", "XXX"),            "XXX1=2=3XXX");
    UTEST_CHECK_EQUAL(nano::join(std::set<int>({ 1, 2, 3 }), ",", nullptr, ")"),             "1,2,3)");
}

UTEST_CASE(strcat)
{
    UTEST_CHECK_EQUAL(nano::strcat(nano::string_t("str"), "x", 'a', 42, nano::string_t("end")), "strxa42end");
    UTEST_CHECK_EQUAL(nano::strcat("str", nano::string_t("x"), 'a', 42, nano::string_t("end")), "strxa42end");
}

UTEST_CASE(contains)
{
    UTEST_CHECK_EQUAL(nano::contains("", 't'), false);
    UTEST_CHECK_EQUAL(nano::contains("text", 't'), true);
    UTEST_CHECK_EQUAL(nano::contains("naNoCv", 't'), false);
    UTEST_CHECK_EQUAL(nano::contains("extension", 't'), true);
}

UTEST_CASE(resize)
{
    UTEST_CHECK_EQUAL(nano::align("text", 10, nano::alignment::left, '='),   "text======");
    UTEST_CHECK_EQUAL(nano::align("text", 10, nano::alignment::right, '='),  "======text");
    UTEST_CHECK_EQUAL(nano::align("text", 10, nano::alignment::left, '='),   "text======");
    UTEST_CHECK_EQUAL(nano::align("text", 10, nano::alignment::center, '='), "===text===");
}

UTEST_CASE(split_str)
{
    const auto str = nano::string_t{"= -token1 token2 something "};
    for (auto tokenizer = nano::tokenizer_t{str, " =-"}; tokenizer; ++ tokenizer)
    {
        switch (tokenizer.count())
        {
        case 1:     UTEST_CHECK_EQUAL(tokenizer.get(), "token1"); break;
        case 2:     UTEST_CHECK_EQUAL(tokenizer.get(), "token2"); break;
        case 3:     UTEST_CHECK_EQUAL(tokenizer.get(), "something"); break;
        default:    UTEST_CHECK(false);
        }
    }
}

UTEST_CASE(split_char)
{
    const auto str = nano::string_t{"= -token1 token2 something "};
    for (auto tokenizer = nano::tokenizer_t{str, "-"}; tokenizer; ++ tokenizer)
    {
        switch (tokenizer.count())
        {
        case 1:     UTEST_CHECK_EQUAL(tokenizer.get(), "= "); break;
        case 2:     UTEST_CHECK_EQUAL(tokenizer.get(), "token1 token2 something "); break;
        default:    UTEST_CHECK(false);
        }
    }
}

UTEST_CASE(split_none)
{
    const auto str = nano::string_t{"= -token1 token2 something "};
    for (auto tokenizer = nano::tokenizer_t{str, "@"}; tokenizer; ++ tokenizer)
    {
        switch (tokenizer.count())
        {
        case 1:     UTEST_CHECK_EQUAL(tokenizer.get(), "= -token1 token2 something "); break;
        default:    UTEST_CHECK(false);
        }
    }
}

UTEST_CASE(lower)
{
    UTEST_CHECK_EQUAL(nano::lower("Token"), "token");
    UTEST_CHECK_EQUAL(nano::lower("ToKEN"), "token");
    UTEST_CHECK_EQUAL(nano::lower("token"), "token");
    UTEST_CHECK_EQUAL(nano::lower("TOKEN"), "token");
    UTEST_CHECK_EQUAL(nano::lower(""), "");
}

UTEST_CASE(upper)
{
    UTEST_CHECK_EQUAL(nano::upper("Token"), "TOKEN");
    UTEST_CHECK_EQUAL(nano::upper("ToKEN"), "TOKEN");
    UTEST_CHECK_EQUAL(nano::upper("token"), "TOKEN");
    UTEST_CHECK_EQUAL(nano::upper("TOKEN"), "TOKEN");
    UTEST_CHECK_EQUAL(nano::upper(""), "");
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

UTEST_CASE(iends_with)
{
    UTEST_CHECK(nano::iends_with("ToKeN", ""));
    UTEST_CHECK(nano::iends_with("ToKeN", "N"));
    UTEST_CHECK(nano::iends_with("ToKeN", "eN"));
    UTEST_CHECK(nano::iends_with("ToKeN", "KeN"));
    UTEST_CHECK(nano::iends_with("ToKeN", "oKeN"));
    UTEST_CHECK(nano::iends_with("ToKeN", "ToKeN"));

    UTEST_CHECK(nano::iends_with("ToKeN", "n"));
    UTEST_CHECK(nano::iends_with("ToKeN", "en"));
    UTEST_CHECK(nano::iends_with("ToKeN", "ken"));
    UTEST_CHECK(nano::iends_with("ToKeN", "oken"));
    UTEST_CHECK(nano::iends_with("ToKeN", "Token"));
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

UTEST_CASE(istarts_with)
{
    UTEST_CHECK(nano::istarts_with("ToKeN", ""));
    UTEST_CHECK(nano::istarts_with("ToKeN", "t"));
    UTEST_CHECK(nano::istarts_with("ToKeN", "to"));
    UTEST_CHECK(nano::istarts_with("ToKeN", "Tok"));
    UTEST_CHECK(nano::istarts_with("ToKeN", "toKe"));
    UTEST_CHECK(nano::istarts_with("ToKeN", "ToKeN"));

    UTEST_CHECK(nano::istarts_with("ToKeN", "t"));
    UTEST_CHECK(nano::istarts_with("ToKeN", "to"));
    UTEST_CHECK(nano::istarts_with("ToKeN", "tok"));
    UTEST_CHECK(nano::istarts_with("ToKeN", "toke"));
    UTEST_CHECK(nano::istarts_with("ToKeN", "Token"));
}

UTEST_CASE(equals)
{
    UTEST_CHECK(!nano::equals("ToKeN", ""));
    UTEST_CHECK(!nano::equals("ToKeN", "N"));
    UTEST_CHECK(!nano::equals("ToKeN", "eN"));
    UTEST_CHECK(!nano::equals("ToKeN", "KeN"));
    UTEST_CHECK(!nano::equals("ToKeN", "oKeN"));
    UTEST_CHECK(nano::equals("ToKeN", "ToKeN"));

    UTEST_CHECK(!nano::equals("ToKeN", "n"));
    UTEST_CHECK(!nano::equals("ToKeN", "en"));
    UTEST_CHECK(!nano::equals("ToKeN", "ken"));
    UTEST_CHECK(!nano::equals("ToKeN", "oken"));
    UTEST_CHECK(!nano::equals("ToKeN", "Token"));
}

UTEST_CASE(iequals)
{
    UTEST_CHECK(!nano::iequals("ToKeN", ""));
    UTEST_CHECK(!nano::iequals("ToKeN", "N"));
    UTEST_CHECK(!nano::iequals("ToKeN", "eN"));
    UTEST_CHECK(!nano::iequals("ToKeN", "KeN"));
    UTEST_CHECK(!nano::iequals("ToKeN", "oKeN"));
    UTEST_CHECK(nano::iequals("ToKeN", "ToKeN"));

    UTEST_CHECK(!nano::iequals("ToKeN", "n"));
    UTEST_CHECK(!nano::iequals("ToKeN", "en"));
    UTEST_CHECK(!nano::iequals("ToKeN", "ken"));
    UTEST_CHECK(!nano::iequals("ToKeN", "oken"));
    UTEST_CHECK(nano::iequals("ToKeN", "Token"));
}

UTEST_CASE(replace_str)
{
    UTEST_CHECK_EQUAL(nano::replace("token-", "en-", "_"), "tok_");
    UTEST_CHECK_EQUAL(nano::replace("t-ken-", "ken", "_"), "t-_-");
}

UTEST_CASE(replace_char)
{
    UTEST_CHECK_EQUAL(nano::replace("token-", '-', '_'), "token_");
    UTEST_CHECK_EQUAL(nano::replace("t-ken-", '-', '_'), "t_ken_");
    UTEST_CHECK_EQUAL(nano::replace("-token", '-', '_'), "_token");
    UTEST_CHECK_EQUAL(nano::replace("token_", '-', '_'), "token_");
}

UTEST_END_MODULE()
