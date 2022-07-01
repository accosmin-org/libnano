#include <nano/core/factory_util.h>
#include <nano/core/strutil.h>
#include <utest/utest.h>

using namespace nano;

struct object_t
{
    object_t()                = default;
    virtual ~object_t()       = default;
    object_t(object_t&&)      = default;
    object_t(const object_t&) = default;
    object_t&   operator=(object_t&&) = default;
    object_t&   operator=(const object_t&) = default;
    virtual int get() const                = 0;
};

template <int tv>
struct objectx_t final : public object_t
{
    objectx_t() = default;

    explicit objectx_t(const int v)
        : m_v(v)
    {
    }

    int get() const override { return m_v; }

    int m_v{tv};
};

using object1_t = objectx_t<1>;
using object2_t = objectx_t<2>;
using object3_t = objectx_t<3>;

template <int tv>
struct factory_traits_t<objectx_t<tv>>
{
    static string_t id() { return scat("id", tv); }

    static string_t description() { return scat("desc", tv); }
};

std::ostream& operator<<(std::ostream& os, const strings_t& strings)
{
    for (size_t i = 0; i < strings.size(); ++i)
    {
        os << strings[i] << (i + 1 == strings.size() ? "" : ",");
    }
    return os;
}

UTEST_BEGIN_MODULE(test_core_factory)

UTEST_CASE(empty)
{
    factory_t<object_t> manager;

    UTEST_CHECK(manager.ids().empty());

    UTEST_CHECK(!manager.has("ds"));
    UTEST_CHECK(!manager.has("ds1"));
    UTEST_CHECK(!manager.has("dd"));
    UTEST_CHECK(!manager.has(""));
    UTEST_CHECK_EQUAL(manager.size(), 0U);
}

UTEST_CASE(retrieval)
{
    factory_t<object_t> manager;

    const string_t id1 = "id1";
    const string_t id2 = "id2";
    const string_t id3 = "id3";

    // register objects
    UTEST_CHECK(manager.add<object1_t>(id1, "desc1"));
    UTEST_CHECK(manager.add<object2_t>(id2, "desc2"));
    UTEST_CHECK(manager.add<object3_t>(id3, "desc3"));
    UTEST_CHECK_EQUAL(manager.size(), 3U);

    // should not be able to register with the same id anymore
    UTEST_CHECK(!manager.add<object1_t>(id1, ""));
    UTEST_CHECK(!manager.add<object2_t>(id1, ""));
    UTEST_CHECK(!manager.add<object3_t>(id1, ""));

    UTEST_CHECK(!manager.add<object1_t>(id2, ""));
    UTEST_CHECK(!manager.add<object2_t>(id2, ""));
    UTEST_CHECK(!manager.add<object3_t>(id2, ""));

    UTEST_CHECK(!manager.add<object1_t>(id3, ""));
    UTEST_CHECK(!manager.add<object2_t>(id3, ""));
    UTEST_CHECK(!manager.add<object3_t>(id3, ""));

    // check retrieval
    UTEST_REQUIRE(manager.has(id1));
    UTEST_REQUIRE(manager.has(id2));
    UTEST_REQUIRE(manager.has(id3));

    UTEST_CHECK(!manager.has(id1 + id2));
    UTEST_CHECK(!manager.has(id2 + id3));
    UTEST_CHECK(!manager.has(id3 + id1));

    const auto object1 = manager.get(id1);
    const auto object2 = manager.get(id2);
    const auto object3 = manager.get(id3);

    UTEST_REQUIRE(object1 != nullptr);
    UTEST_REQUIRE(object2 != nullptr);
    UTEST_REQUIRE(object3 != nullptr);

    UTEST_CHECK_EQUAL(object1->get(), 1);
    UTEST_CHECK_EQUAL(object2->get(), 2);
    UTEST_CHECK_EQUAL(object3->get(), 3);

    UTEST_CHECK(manager.get("") == nullptr);
    UTEST_CHECK(manager.get(id1 + id2 + "ddd") == nullptr);
    UTEST_CHECK(manager.get("not there") == nullptr);

    // check retrieval by regex
    const auto ids0   = strings_t{};
    const auto ids1   = strings_t{id1};
    const auto ids12  = strings_t{id1, id2};
    const auto ids123 = strings_t{id1, id2, id3};
    UTEST_CHECK_EQUAL(manager.ids(), ids123);
    UTEST_CHECK_EQUAL(manager.ids(std::regex("[a-z]+[0-9]")), ids123);
    UTEST_CHECK_EQUAL(manager.ids(std::regex("[a-z]+1")), ids1);
    UTEST_CHECK_EQUAL(manager.ids(std::regex(".+")), ids123);
    UTEST_CHECK_EQUAL(manager.ids(std::regex("id1")), ids1);
    UTEST_CHECK_EQUAL(manager.ids(std::regex("id[0-9]")), ids123);
    UTEST_CHECK_EQUAL(manager.ids(std::regex("id[1|2]")), ids12);
    UTEST_CHECK_EQUAL(manager.ids(std::regex("id7")), ids0);
    UTEST_CHECK_EQUAL(manager.ids(std::regex("id1|id2|id4")), ids12);
}

UTEST_CASE(retrieval_default)
{
    factory_t<object_t> manager;

    const string_t id1 = factory_traits_t<object1_t>::id();
    const string_t id2 = factory_traits_t<object2_t>::id();
    const string_t id3 = factory_traits_t<object3_t>::id();

    // register objects
    UTEST_CHECK(manager.add_by_type<object1_t>(7));
    UTEST_CHECK(manager.add_by_type<object2_t>());
    UTEST_CHECK(manager.add_by_type<object3_t>(5));
    UTEST_CHECK_EQUAL(manager.size(), 3U);

    // check retrieval with the default arguments
    UTEST_REQUIRE(manager.has(id1));
    UTEST_REQUIRE(manager.has(id2));
    UTEST_REQUIRE(manager.has(id3));

    const auto object1 = manager.get(id1);
    const auto object2 = manager.get(id2);
    const auto object3 = manager.get(id3);

    UTEST_REQUIRE(object1 != nullptr);
    UTEST_REQUIRE(object2 != nullptr);
    UTEST_REQUIRE(object3 != nullptr);

    UTEST_CHECK_EQUAL(object1->get(), 7);
    UTEST_CHECK_EQUAL(object2->get(), 2);
    UTEST_CHECK_EQUAL(object3->get(), 5);

    UTEST_CHECK_EQUAL(manager.description(id1), factory_traits_t<object1_t>::description());
    UTEST_CHECK_EQUAL(manager.description(id2), factory_traits_t<object2_t>::description());
    UTEST_CHECK_EQUAL(manager.description(id3), factory_traits_t<object3_t>::description());
    UTEST_CHECK_EQUAL(manager.description("none"), "");
}

UTEST_CASE(make_object_table)
{
    factory_t<object_t> manager;

    UTEST_CHECK(manager.add<object1_t>("id1", "desc1"));
    UTEST_CHECK(manager.add<object2_t>("id2", "desc2"));
    UTEST_CHECK(manager.add<object3_t>("id3", "desc3"));

    const auto table = make_table("object", manager);
    UTEST_CHECK_EQUAL(scat(table), "|--------|-------------|\n"
                                   "| object | description |\n"
                                   "|--------|-------------|\n"
                                   "| id1    | desc1       |\n"
                                   "| id2    | desc2       |\n"
                                   "| id3    | desc3       |\n"
                                   "|--------|-------------|\n");
}

UTEST_END_MODULE()
