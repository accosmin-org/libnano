#include <utest/utest.h>
#include "core/factory.h"

using namespace nano;

struct object_t
{
        object_t() = default;
        virtual ~object_t() = default;
        virtual int get() const = 0;
};

template <int tv>
struct objectx_t final : public object_t
{
        objectx_t() = default;
        explicit objectx_t(const int v) : m_v(v) {}
        int get() const override { return m_v; }
        int m_v{tv};
};

using object1_t = objectx_t<1>;
using object2_t = objectx_t<2>;
using object3_t = objectx_t<3>;

std::ostream& operator<<(std::ostream& os, const strings_t& strings)
{
        for (size_t i = 0; i < strings.size(); ++ i)
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
}

UTEST_CASE(retrieval)
{
        factory_t<object_t, int> manager;

        const string_t id1 = "obj1";
        const string_t id2 = "obj2";
        const string_t id3 = "obj3";

        // register objects
        UTEST_CHECK(manager.add<object1_t>(id1, "test obj1"));
        UTEST_CHECK(manager.add<object2_t>(id2, "test obj2"));
        UTEST_CHECK(manager.add<object3_t>(id3, "test obj3"));

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

        UTEST_REQUIRE(manager.get(id1, 0) != nullptr);
        UTEST_REQUIRE(manager.get(id2, 0) != nullptr);
        UTEST_REQUIRE(manager.get(id3, 0) != nullptr);

        UTEST_CHECK_EQUAL(manager.get(id1, 1)->get(), 1);
        UTEST_CHECK_EQUAL(manager.get(id2, 2)->get(), 2);
        UTEST_CHECK_EQUAL(manager.get(id3, 3)->get(), 3);

        UTEST_CHECK_EQUAL(manager.get(id1, 42)->get(), 42);
        UTEST_CHECK_EQUAL(manager.get(id2, 42)->get(), 42);
        UTEST_CHECK_EQUAL(manager.get(id3, 42)->get(), 42);

        UTEST_CHECK(manager.get("", 0) == nullptr);
        UTEST_CHECK(manager.get(id1 + id2 + "ddd", 0) == nullptr);
        UTEST_CHECK(manager.get("not there", 0) == nullptr);

        // check retrieval by regex
        const auto ids0 = strings_t{};
        const auto ids1 = strings_t{id1};
        const auto ids12 = strings_t{id1, id2};
        const auto ids123 = strings_t{id1, id2, id3};
        UTEST_CHECK_EQUAL(manager.ids(), ids123);
        UTEST_CHECK_EQUAL(manager.ids(std::regex("[a-z]+[0-9]")), ids123);
        UTEST_CHECK_EQUAL(manager.ids(std::regex("[a-z]+1")), ids1);
        UTEST_CHECK_EQUAL(manager.ids(std::regex(".+")), ids123);
        UTEST_CHECK_EQUAL(manager.ids(std::regex("obj1")), ids1);
        UTEST_CHECK_EQUAL(manager.ids(std::regex("obj[0-9]")), ids123);
        UTEST_CHECK_EQUAL(manager.ids(std::regex("obj[1|2]")), ids12);
        UTEST_CHECK_EQUAL(manager.ids(std::regex("obj7")), ids0);
}

UTEST_CASE(retrieval_default)
{
        factory_t<object_t> manager;

        const string_t id1 = "obj1";
        const string_t id2 = "obj2";
        const string_t id3 = "obj3";

        // register objects
        UTEST_CHECK(manager.add<object1_t>(id1, "test obj1"));
        UTEST_CHECK(manager.add<object2_t>(id2, "test obj2"));
        UTEST_CHECK(manager.add<object3_t>(id3, "test obj3"));

        // check retrieval with the default arguments
        UTEST_REQUIRE(manager.has(id1));
        UTEST_REQUIRE(manager.has(id2));
        UTEST_REQUIRE(manager.has(id3));

        UTEST_REQUIRE(manager.get(id1) != nullptr);
        UTEST_REQUIRE(manager.get(id2) != nullptr);
        UTEST_REQUIRE(manager.get(id3) != nullptr);

        UTEST_CHECK_EQUAL(manager.get(id1)->get(), 1);
        UTEST_CHECK_EQUAL(manager.get(id2)->get(), 2);
        UTEST_CHECK_EQUAL(manager.get(id3)->get(), 3);
}

UTEST_END_MODULE()
