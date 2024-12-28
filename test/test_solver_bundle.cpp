#include <fixture/solver.h>

using namespace nano;

namespace
{
auto make_solvers(const tensor_size_t bundle_max_size = 10)
{
    auto solvers = rsolvers_t{};
    solvers.emplace_back(make_solver("ellipsoid"));
    for (const auto* const solver_id : {"rqb", "fpba1", "fpba2"})
    {
        auto solver                                                          = make_solver(solver_id);
        solver->parameter(scat("solver::", solver_id, "::bundle::max_size")) = bundle_max_size;
        solvers.emplace_back(std::move(solver));
    }
    return solvers;
}
} // namespace

UTEST_BEGIN_MODULE(test_solver_bundle)

UTEST_CASE(smooth_bundle)
{
    check_solvers_on_smooth_functions(make_solvers());
}

UTEST_CASE(nonsmooth_bundle)
{
    check_solvers_on_nonsmooth_functions(make_solvers());
}

UTEST_END_MODULE()
