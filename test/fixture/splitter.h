#include <nano/splitter.h>
#include <utest/utest.h>

using namespace nano;

static auto make_splitter(const string_t& name, const tensor_size_t folds, const uint64_t seed = 42U)
{
    auto splitter = splitter_t::all().get(name);
    UTEST_REQUIRE(splitter);
    splitter->parameter("splitter::seed")  = seed;
    splitter->parameter("splitter::folds") = folds;
    return splitter;
}
