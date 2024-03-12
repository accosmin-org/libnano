#include "fixture/solver.h"

using namespace nano;

inline auto make_solver_ids()
{
    return strings_t{"ellipsoid", "rqb", "fpba1", "fpba2"};
}

inline auto make_solvers(const char* const pruning, const tensor_size_t bundle_max_size = 10)
{
    auto solvers = rsolvers_t{};
    solvers.emplace_back(make_solver("ellipsoid"));
    for (const auto* const solver_id : {"rqb", "fpba1", "fpba2"})
    {
        auto solver                                                          = make_solver(solver_id);
        solver->parameter(scat("solver::", solver_id, "::bundle::pruning"))  = pruning;
        solver->parameter(scat("solver::", solver_id, "::bundle::max_size")) = bundle_max_size;
        solvers.emplace_back(std::move(solver));
    }
    return solvers;
}

UTEST_BEGIN_MODULE(test_solver_bundle)

UTEST_CASE(smooth_bundle_oldest)
{
    check_solvers_on_smooth_functions(make_solvers("oldest"));
}

UTEST_CASE(smooth_bundle_largest)
{
    check_solvers_on_smooth_functions(make_solvers("largest"));
}

UTEST_CASE(nonsmooth_bundle_oldest)
{
    check_solvers_on_nonsmooth_functions(make_solvers("oldest"));
}

UTEST_CASE(nonsmooth_bundle_largest)
{
    check_solvers_on_nonsmooth_functions(make_solvers("largest"));
}

UTEST_END_MODULE()
