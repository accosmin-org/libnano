#include <nano/splitter.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] inline auto make_splitter(const string_t& name = "k-fold", const int folds = 2)
{
    auto splitter = splitter_t::all().get(name);
    UTEST_REQUIRE(splitter);
    splitter->parameter("splitter::folds") = folds;
    return splitter;
}
