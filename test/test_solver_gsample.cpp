#include <fixture/solver.h>

using namespace nano;

namespace
{
[[maybe_unused]] auto make_solver_ids()
{
    return strings_t{"ellipsoid", "gs", "gs-lbfgs", "ags", "ags-lbfgs"};
}
} // namespace

UTEST_BEGIN_MODULE(test_solver_gsample)

UTEST_CASE(smooth)
{
    // FIXME: enable the test when the gradient sampling solvers are more robust!
    // check_minimize(make_solver_ids(), function_t::make({4, 4, function_type::convex_smooth}));
}

UTEST_CASE(nonsmooth)
{
    // FIXME: enable the test when the gradient sampling solvers are more robust!
    // check_minimize(make_solver_ids(), function_t::make({4, 4, function_type::convex}));
}

UTEST_END_MODULE()
