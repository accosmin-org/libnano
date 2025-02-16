#include <fixture/solver.h>

using namespace nano;

namespace
{
auto make_solver_ids()
{
    return strings_t{"ellipsoid", "sgm", "cocob", "sda", "wda", "pgm", "dgm", "fgm", "asga2", "asga4", "osga"};
}
} // namespace

UTEST_BEGIN_MODULE(test_solver_nonlinear)

UTEST_CASE(smooth)
{
    check_minimize(make_solver_ids(), function_t::make({4, 4, function_type::convex_smooth}));
}

UTEST_CASE(nonsmooth)
{
    check_minimize(make_solver_ids(), function_t::make({4, 4, function_type::convex}));
}

UTEST_END_MODULE()
