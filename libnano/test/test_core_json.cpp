#include <nano/json.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_json)

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

UTEST_END_MODULE()
