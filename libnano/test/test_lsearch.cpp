#include <utest/utest.h>
#include <nano/solver.h>
#include <nano/numeric.h>

using namespace nano;

static auto get_state0(const rfunction_t& function)
{
    auto state0 = solver_state_t{*function, vector_t::Random(function->size())};
    state0.d = -state0.g;
    return state0;
}

UTEST_BEGIN_MODULE(test_lsearch)

UTEST_CASE(strategy_backtrack)
{
    const auto lsearch = get_lsearch_strategies().get("backtrack");
    lsearch->c1(scalar_t{1e-4});
    lsearch->c2(scalar_t{9e-1});
    lsearch->max_iterations(100);

    // check that the line-search doesn't fail from various starting points
    //  and the resulting point satisfies the Armijo condition
    for (const auto& function : get_functions(1, 4, std::regex(".+")))
    {
        for (auto i = 0; i < 100; ++ i)
        {
            const auto t0 = scalar_t{1};
            const auto state0 = get_state0(function);

            solver_state_t state = state0;
            UTEST_CHECK(lsearch->get(state0, t0, state));
            UTEST_CHECK(state0.has_armijo(state0, lsearch->c1()));
        }
    }
}

UTEST_END_MODULE()
