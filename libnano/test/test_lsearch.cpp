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

static auto get_lsearch(const string_t& id, const scalar_t c1 = 1e-4, const scalar_t c2 = 9e-1)
{
    auto lsearch = lsearch_strategy_t::all().get(id);
    UTEST_REQUIRE(lsearch);
    lsearch->c1(c1);
    lsearch->c2(c2);
    lsearch->max_iterations(100);
    return lsearch;
}

static void set_logger(const rlsearch_strategy_t& lsearch, const string_t& function_name, const string_t& lsearch_id,
    const scalar_t t0, const solver_state_t& state0, std::stringstream& stream)
{
    stream
        << std::fixed << std::setprecision(8) << function_name << " " << lsearch_id
        << ": x0=[" << state0.x.transpose() << "],t0=" << t0 << ",f0=" << state0.f << "\n";
    lsearch->logger([&] (const solver_state_t& state)
    {
        stream
            << "\tt=" << state.t << ",f=" << state.f << ",g=" << state.convergence_criterion()
            << ",armijo=" << state.has_armijo(state0, lsearch->c1())
            << ",wolfe=" << state.has_wolfe(state0, lsearch->c2())
            << ",swolfe=" << state.has_strong_wolfe(state0, lsearch->c2())
            << ",awolfe=" << state.has_approx_wolfe(state0, lsearch->c1(), lsearch->c2()) << ".\n";
    });
}

// todo: extend the checks for different c1 and c2 values (1e-1+9e-1, 1e-4+1e-1, 1e-4+9e-1)

UTEST_BEGIN_MODULE(test_lsearch)

UTEST_CASE(backtrack)
{
    const auto lsearch = get_lsearch("backtrack");

    // check that the line-search doesn't fail from various starting points
    //  and the resulting point satisfies the Armijo condition
    for (const auto& function : get_functions(1, 16, std::regex(".+")))
    {
        for (auto i = 0; i < 10; ++ i)
        {
            const auto t0 = scalar_t{1};
            const auto state0 = get_state0(function);
            const auto old_n_failures = n_failures.load();

            std::stringstream stream;
            set_logger(lsearch, function->name(), "backtrack", t0, state0, stream);

            solver_state_t state = state0;
            UTEST_CHECK(lsearch->get(state0, t0, state));
            UTEST_CHECK(state);
            UTEST_CHECK(state0);
            UTEST_CHECK(state.has_armijo(state0, lsearch->c1()));

            if (old_n_failures != n_failures.load())
            {
                std::cout << stream.str();
            }
        }
    }
}

UTEST_CASE(lemarechal)
{
    const auto lsearch = get_lsearch("lemarechal");

    // check that the line-search doesn't fail from various starting points
    //  and the resulting point satisfies the Armijo and the (non-strong) Wolfe conditions
    for (const auto& function : get_functions(1, 16, std::regex(".+")))
    {
        for (auto i = 0; i < 10; ++ i)
        {
            const auto t0 = scalar_t{1};
            const auto state0 = get_state0(function);
            const auto old_n_failures = n_failures.load();

            std::stringstream stream;
            set_logger(lsearch, function->name(), "lemarechal", t0, state0, stream);

            auto state = state0;
            UTEST_CHECK(lsearch->get(state0, t0, state));
            UTEST_CHECK(state);
            UTEST_CHECK(state0);
            UTEST_CHECK(state.has_armijo(state0, lsearch->c1()));
            UTEST_CHECK(state.has_wolfe(state0, lsearch->c2()));

            if (old_n_failures != n_failures.load())
            {
                std::cout << stream.str();
            }
        }
    }
}

UTEST_CASE(morethuente)
{
    const auto lsearch = get_lsearch("morethuente");

    // check that the line-search doesn't fail from various starting points
    //  and the resulting point satisfies the Armijo and the strong Wolfe conditions
    for (const auto& function : get_functions(1, 16, std::regex(".+")))
    {
        for (auto i = 0; i < 10; ++ i)
        {
            const auto t0 = scalar_t{1};
            const auto state0 = get_state0(function);
            const auto old_n_failures = n_failures.load();

            std::stringstream stream;
            set_logger(lsearch, function->name(), "morethuente", t0, state0, stream);

            solver_state_t state = state0;
            UTEST_CHECK(lsearch->get(state0, t0, state));
            UTEST_CHECK(state);
            UTEST_CHECK(state0);
            UTEST_CHECK(state.has_armijo(state0, lsearch->c1()));
            UTEST_CHECK(state.has_strong_wolfe(state0, lsearch->c2()));

            if (old_n_failures != n_failures.load())
            {
                std::cout << stream.str();
            }
        }
    }
}

UTEST_CASE(nocedalwright)
{
    const auto lsearch = get_lsearch("nocedalwright");

    // check that the line-search doesn't fail from various starting points
    //  and the resulting point satisfies the Armijo and the strong Wolfe conditions
    for (const auto& function : get_functions(1, 16, std::regex(".+")))
    {
        for (auto i = 0; i < 10; ++ i)
        {
            const auto t0 = scalar_t{1};
            const auto state0 = get_state0(function);
            const auto old_n_failures = n_failures.load();

            std::stringstream stream;
            set_logger(lsearch, function->name(), "nocedalwright", t0, state0, stream);

            solver_state_t state = state0;
            UTEST_CHECK(lsearch->get(state0, t0, state));
            UTEST_CHECK(state);
            UTEST_CHECK(state0);
            UTEST_CHECK(state.has_armijo(state0, lsearch->c1()));
            UTEST_CHECK(state.has_strong_wolfe(state0, lsearch->c2()));

            if (old_n_failures != n_failures.load())
            {
                std::cout << stream.str();
            }
        }
    }
}

UTEST_CASE(cgdescent)
{
    const auto lsearch = get_lsearch("cgdescent");
    const auto epsilon = 1e-6;// todo: get the updated value of epsilon!!!

    // check that the line-search doesn't fail from various starting points
    //  and the resulting point satisfies the Armijo and the approximated Wolfe conditions
    for (const auto& function : get_functions(1, 16, std::regex(".+")))
    {
        for (auto i = 0; i < 10; ++ i)
        {
            const auto t0 = scalar_t{1};
            const auto state0 = get_state0(function);
            const auto old_n_failures = n_failures.load();

            std::stringstream stream;
            set_logger(lsearch, function->name(), "cgdescent", t0, state0, stream);

            solver_state_t state = state0;
            UTEST_CHECK(lsearch->get(state0, t0, state));
            UTEST_CHECK(state);
            UTEST_CHECK(state0);
            UTEST_CHECK(
                (state.has_armijo(state0, lsearch->c1()) && state.has_wolfe(state0, lsearch->c1())) ||
                (state.has_approx_armijo(state0, epsilon) && state.has_approx_wolfe(state0, lsearch->c1(), lsearch->c2())));

            if (old_n_failures != n_failures.load())
            {
                std::cout << stream.str();
            }
        }
    }
}

UTEST_END_MODULE()
