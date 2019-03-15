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

static auto get_lsearch(const string_t& id)
{
    std::cout << "evaluating lsearch method " << id << "...\n";
    auto lsearch = lsearch_strategy_t::all().get(id);
    UTEST_REQUIRE(lsearch);
    lsearch->c1(scalar_t{1e-4});
    lsearch->c2(scalar_t{9e-1});
    lsearch->max_iterations(100);
    return lsearch;
}

static void set_logger(const rlsearch_strategy_t& lsearch, const scalar_t t0, const solver_state_t& state0)
{
    std::cout << std::fixed << std::setprecision(8) << "....x0=" << state0.x.transpose() << "\n";
    lsearch->logger([&lsearch = lsearch, t0 = t0, &state0 = state0] (const solver_state_t& state)
    {
        std::cout << std::fixed << std::setprecision(8) << "........"
            << "t0=" << t0 << ",f0=" << state0.f
            << ",t=" << state.t << ",f=" << state.f
            << ",g=" << state.convergence_criterion()
            << ",armijo=" << state.has_armijo(state0, lsearch->c1())
            << ",wolfe=" << state.has_wolfe(state0, lsearch->c2())
            << ",swolfe=" << state.has_strong_wolfe(state0, lsearch->c2())
            << ",awolfe=" << state.has_approx_wolfe(state0, lsearch->c1(), lsearch->c2()) << "\n";
    });
}

// todo: extend the checks for different c1 and c2 values (1e-1+9e-1, 1e-4+1e-1, 1e-4+9e-1)
// todo: check configuring line-search strategies
// todo: verbose only when a failure is detected - add support for this in utest

UTEST_BEGIN_MODULE(test_lsearch)

UTEST_CASE(backtrack)
{
    const auto lsearch = get_lsearch("backtrack");

    // check that the line-search doesn't fail from various starting points
    //  and the resulting point satisfies the Armijo condition
    for (const auto& function : get_functions(1, 4, std::regex(".+")))
    {
        std::cout << ">>evaluating function " << function->name() << "...\n";

        for (auto i = 0; i < 100; ++ i)
        {
            const auto t0 = scalar_t{1};
            const auto state0 = get_state0(function);
            set_logger(lsearch, t0, state0);

            solver_state_t state = state0;
            UTEST_CHECK(lsearch->get(state0, t0, state));
            UTEST_CHECK(state);
            UTEST_CHECK(state0);
            UTEST_CHECK(state.has_armijo(state0, lsearch->c1()));
        }
    }
}

UTEST_CASE(lemarechal)
{
    const auto lsearch = get_lsearch("lemarechal");

    // check that the line-search doesn't fail from various starting points
    //  and the resulting point satisfies the Armijo and the (non-strong) Wolfe conditions
    for (const auto& function : get_functions(1, 4, std::regex(".+")))
    {
        std::cout << ">>evaluating function " << function->name() << "...\n";

        for (auto i = 0; i < 100; ++ i)
        {
            const auto t0 = scalar_t{1};
            const auto state0 = get_state0(function);
            set_logger(lsearch, t0, state0);

            solver_state_t state = state0;
            UTEST_CHECK(lsearch->get(state0, t0, state));
            UTEST_CHECK(state);
            UTEST_CHECK(state0);
            UTEST_CHECK(state.has_armijo(state0, lsearch->c1()));
            UTEST_CHECK(state.has_wolfe(state0, lsearch->c2()));
        }
    }
}

UTEST_CASE(morethuente)
{
    const auto lsearch = get_lsearch("morethuente");

    // check that the line-search doesn't fail from various starting points
    //  and the resulting point satisfies the Armijo and the strong Wolfe conditions
    for (const auto& function : get_functions(1, 4, std::regex(".+")))
    {
        std::cout << ">>evaluating function " << function->name() << "...\n";

        for (auto i = 0; i < 100; ++ i)
        {
            const auto t0 = scalar_t{1};
            const auto state0 = get_state0(function);
            set_logger(lsearch, t0, state0);

            solver_state_t state = state0;
            UTEST_CHECK(lsearch->get(state0, t0, state));
            UTEST_CHECK(state);
            UTEST_CHECK(state0);
            UTEST_CHECK(state.has_armijo(state0, lsearch->c1()));
            UTEST_CHECK(state.has_strong_wolfe(state0, lsearch->c2()));
        }
    }
}

UTEST_CASE(nocedalwright)
{
    const auto lsearch = get_lsearch("nocedalwright");

    // check that the line-search doesn't fail from various starting points
    //  and the resulting point satisfies the Armijo and the strong Wolfe conditions
    for (const auto& function : get_functions(1, 4, std::regex(".+")))
    {
        std::cout << ">>evaluating function " << function->name() << "...\n";

        for (auto i = 0; i < 100; ++ i)
        {
            const auto t0 = scalar_t{1};
            const auto state0 = get_state0(function);
            set_logger(lsearch, t0, state0);

            solver_state_t state = state0;
            UTEST_CHECK(lsearch->get(state0, t0, state));
            UTEST_CHECK(state);
            UTEST_CHECK(state0);
            UTEST_CHECK(state.has_armijo(state0, lsearch->c1()));
            UTEST_CHECK(state.has_strong_wolfe(state0, lsearch->c2()));
        }
    }
}

UTEST_CASE(cgdescent)
{
    const auto lsearch = get_lsearch("cgdescent");
    const auto epsilon = 1e-6;// todo: get the updated value of epsilon!!!

    // check that the line-search doesn't fail from various starting points
    //  and the resulting point satisfies the Armijo and the approximated Wolfe conditions
    for (const auto& function : get_functions(1, 4, std::regex(".+")))
    {
        std::cout << ">>evaluating function " << function->name() << "...\n";

        for (auto i = 0; i < 100; ++ i)
        {
            const auto t0 = scalar_t{1};
            const auto state0 = get_state0(function);
            set_logger(lsearch, t0, state0);

            solver_state_t state = state0;
            UTEST_CHECK(lsearch->get(state0, t0, state));
            UTEST_CHECK(state);
            UTEST_CHECK(state0);
            UTEST_CHECK(
                (state.has_armijo(state0, lsearch->c1()) && state.has_wolfe(state0, lsearch->c1())) ||
                (state.has_approx_armijo(state0, epsilon) && state.has_approx_wolfe(state0, lsearch->c1(), lsearch->c2())));
        }
    }
}

UTEST_END_MODULE()
