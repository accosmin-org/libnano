#include <nano/json.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_json)

UTEST_CASE(to_json_empty)
{
    json_t json;
    UTEST_CHECK_NOTHROW(to_json(json));
    UTEST_CHECK_EQUAL(json.dump(), "null");
}

UTEST_CASE(to_json)
{
    json_t json;
    UTEST_CHECK_NOTHROW(to_json(json, "str", "string", "int", 1, "float", 1.0f));
    UTEST_CHECK_EQUAL(json.dump(), "{\"float\":\"1\",\"int\":\"1\",\"str\":\"string\"}");
}

UTEST_CASE(from_json)
{
    json_t json;
    json["str"] = "string";
    json["float"] = 1.0f;
    json["int"] = 1;
    string_t string;
    int integer = -1;
    float floating = -1.0f;
    UTEST_CHECK_NOTHROW(from_json(json, "str", string, "int", integer, "float", floating));
    UTEST_CHECK_EQUAL(string, "string");
    UTEST_CHECK_EQUAL(integer, 1);
    UTEST_CHECK_EQUAL(floating, 1.0f);
}

UTEST_CASE(from_range_range_ok)
{
    const auto json = to_json("value", 1);

    int value = 0;
    UTEST_CHECK_NOTHROW(from_json_range(json, "value", value, -1, +10));
    UTEST_CHECK_EQUAL(value, 1);
}

UTEST_CASE(from_range_range_nok)
{
    const auto json = to_json("value", 1);

    int value = 0;
    UTEST_CHECK_THROW(from_json_range(json, "value", value, +2, +10), std::invalid_argument);
    UTEST_CHECK_EQUAL(value, 1);
}

UTEST_CASE(from_range_range_invalid)
{
    const auto json = to_json("value", "this-is-not-a-valid-integer");

    int value = 0;
    UTEST_CHECK_THROW(from_json_range(json, "value", value, +5, +10), std::exception);
    UTEST_CHECK_EQUAL(value, 0);
}

UTEST_CASE(from_range_range_missing)
{
    const auto json = to_json("valuex", 3);

    int value = 0;
    UTEST_CHECK_NOTHROW(from_json_range(json, "value", value, +5, +10));
    UTEST_CHECK_EQUAL(value, 0);
}

UTEST_END_MODULE()
