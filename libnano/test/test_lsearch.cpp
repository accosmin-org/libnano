#include <iomanip>
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
        std::cout << "evaluating function " << function->name() << "...\n";

        for (auto i = 0; i < 10; ++ i)
        {
            const auto t0 = scalar_t{1};
            const auto state0 = get_state0(function);

            std::cout << std::setprecision(8) << "..x0=" << state0.x.transpose() << "\n";
            lsearch->logger([&] (const solver_state_t& state)
            {
                std::cout << std::setprecision(8) << "    "
                    << "t0=" << t0 << ",f0=" << state0.f
                    << ",t=" << state.t << ",f=" << state.f
                    << ",g=" << state.convergence_criterion()
                    << ",armijo=" << state.has_armijo(state0, lsearch->c1()) << "\n";
            });

            solver_state_t state = state0;
            UTEST_CHECK(lsearch->get(state0, t0, state));
            UTEST_CHECK(state.has_armijo(state0, lsearch->c1()));
        }
    }
}

UTEST_END_MODULE()
