#include "fixture/solver.h"

using namespace nano;

namespace
{
auto make_solver_ids()
{
    return strings_t{"ellipsoid", "sgm", "cocob", "sda", "wda", "pgm", "dgm", "fgm", "asga2", "asga4"};
}
} // namespace

UTEST_BEGIN_MODULE(test_solver_unreliable)

UTEST_CASE(smooth)
{
    check_solvers_on_smooth_functions(make_solver_ids());
}

UTEST_CASE(nonsmooth)
{
    check_solvers_on_nonsmooth_functions(make_solver_ids());
}

UTEST_END_MODULE()
