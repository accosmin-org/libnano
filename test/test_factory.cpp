#include <nano/core/factory_util.h>
#include <nano/core/strutil.h>
#include <nano/lsearch0.h>
#include <utest/utest.h>

using namespace nano;

namespace
{
class object_t final : public typed_t, public clonable_t<object_t>
{
public:
    explicit object_t(const int tv)
        : typed_t(scat("id", tv))
        , m_tv(tv)
    {
    }

    ~object_t() override                 = default;
    object_t(object_t&&)                 = default;
    object_t(const object_t&)            = default;
    object_t& operator=(object_t&&)      = default;
    object_t& operator=(const object_t&) = default;

    int get() const { return m_tv; }

    std::unique_ptr<object_t> clone() const override { return std::make_unique<object_t>(*this); }

private:
    int m_tv{0};
};

std::ostream& operator<<(std::ostream& stream, const strings_t& strings)
{
    for (size_t i = 0; i < strings.size(); ++i)
    {
        stream << strings[i] << (i + 1 == strings.size() ? "" : ",");
    }
    return stream;
}
} // namespace

UTEST_BEGIN_MODULE(test_factory)

UTEST_CASE(empty)
{
    const auto manager = factory_t<object_t>{};

    UTEST_CHECK(manager.ids().empty());

    UTEST_CHECK(!manager.has("ds"));
    UTEST_CHECK(!manager.has("ds1"));
    UTEST_CHECK(!manager.has("dd"));
    UTEST_CHECK(!manager.has(""));
    UTEST_CHECK_EQUAL(manager.size(), 0U);
}

UTEST_CASE(retrieval)
{
    auto manager = factory_t<object_t>{};

    const auto id1 = scat("id", 1);
    const auto id2 = scat("id", 2);
    const auto id3 = scat("id", 3);

    // register objects
    UTEST_CHECK(manager.add<object_t>("desc1", 1));
    UTEST_CHECK(manager.add<object_t>("desc2", 2));
    UTEST_CHECK(manager.add<object_t>("desc3", 3));
    UTEST_CHECK_EQUAL(manager.size(), 3U);

    // should not be able to register with the same id anymore
    UTEST_CHECK(!manager.add<object_t>("", 1));
    UTEST_CHECK(!manager.add<object_t>("", 2));
    UTEST_CHECK(!manager.add<object_t>("", 3));

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

    UTEST_CHECK_EQUAL(manager.description(id1), "desc1");
    UTEST_CHECK_EQUAL(manager.description(id2), "desc2");
    UTEST_CHECK_EQUAL(manager.description(id3), "desc3");
    UTEST_CHECK_EQUAL(manager.description("none"), "");
}

UTEST_CASE(make_object_table)
{
    auto manager = factory_t<object_t>{};

    UTEST_CHECK(manager.add<object_t>("desc1", 1));
    UTEST_CHECK(manager.add<object_t>("desc2", 2));
    UTEST_CHECK(manager.add<object_t>("desc3", 3));

    const auto table = make_table("object", manager);
    UTEST_CHECK_EQUAL(scat(table), "|--------|-------------|\n"
                                   "| object | description |\n"
                                   "|--------|-------------|\n"
                                   "| id1    | desc1       |\n"
                                   "| id2    | desc2       |\n"
                                   "| id3    | desc3       |\n"
                                   "|--------|-------------|\n");
}

UTEST_CASE(make_table_with_params_one)
{
    const auto table = make_table_with_params("lsearch0", lsearch0_t::all(), "what?!");
    UTEST_CHECK_EQUAL(scat(table), "|----------|-----------|-------|--------|\n"
                                   "| lsearch0 | parameter | value | domain |\n"
                                   "|----------|-----------|-------|--------|\n");
}

UTEST_CASE(make_table_with_params_some)
{
    const auto table = make_table_with_params("lsearch0", lsearch0_t::all(), "linear|quadratic");
    UTEST_CHECK_EQUAL(scat(table), "|-----------|-------------------------------|----------|---------------------|\n"
                                   "| lsearch0  | parameter                     | value    | domain              |\n"
                                   "|-----------|-------------------------------|----------|---------------------|\n"
                                   "| linear    | linearly interpolate the previous line-search step size        |\n"
                                   "|-----------|-------------------------------|----------|---------------------|\n"
                                   "| |...      | lsearch0::epsilon             | 1e-06    | 0 < 1e-06 < 1       |\n"
                                   "| |...      | lsearch0::linear::beta        | 10       | 1 < 10 < 1e+06      |\n"
                                   "| |...      | lsearch0::linear::alpha       | 1.01     | 1 < 1.01 < 1e+06    |\n"
                                   "|-----------|-------------------------------|----------|---------------------|\n"
                                   "| quadratic | quadratically interpolate the previous line-search step size   |\n"
                                   "|-----------|-------------------------------|----------|---------------------|\n"
                                   "| |...      | lsearch0::epsilon             | 1e-06    | 0 < 1e-06 < 1       |\n"
                                   "| |...      | lsearch0::quadratic::beta     | 10       | 1 < 10 < 1e+06      |\n"
                                   "| |...      | lsearch0::quadratic::alpha    | 1.01     | 1 < 1.01 < 1e+06    |\n"
                                   "|-----------|-------------------------------|----------|---------------------|\n");
}

UTEST_CASE(make_table_with_params_only)
{
    const auto table = make_table_with_params("lsearch0", lsearch0_t::all(), "quadratic|what?!");
    UTEST_CHECK_EQUAL(scat(table), "|-----------|-------------------------------|----------|---------------------|\n"
                                   "| lsearch0  | parameter                     | value    | domain              |\n"
                                   "|-----------|-------------------------------|----------|---------------------|\n"
                                   "| quadratic | quadratically interpolate the previous line-search step size   |\n"
                                   "|-----------|-------------------------------|----------|---------------------|\n"
                                   "| |...      | lsearch0::epsilon             | 1e-06    | 0 < 1e-06 < 1       |\n"
                                   "| |...      | lsearch0::quadratic::beta     | 10       | 1 < 10 < 1e+06      |\n"
                                   "| |...      | lsearch0::quadratic::alpha    | 1.01     | 1 < 1.01 < 1e+06    |\n"
                                   "|-----------|-------------------------------|----------|---------------------|\n");
}

UTEST_END_MODULE()
