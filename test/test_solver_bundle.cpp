#include "fixture/solver.h"

using namespace nano;

inline auto make_solver_ids()
{
    return strings_t{"ellipsoid", "rqb", "fpba1", "fpba2"};
}

UTEST_BEGIN_MODULE(test_solver_bundle)

UTEST_CASE(smooth)
{
    check_solvers_on_smooth_functions(make_solver_ids());
}

UTEST_CASE(nonsmooth)
{
    check_solvers_on_nonsmooth_functions(make_solver_ids());
}

UTEST_END_MODULE()
