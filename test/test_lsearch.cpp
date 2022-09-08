#include "fixture/function.h"
#include <iomanip>
#include <nano/lsearchk/backtrack.h>
#include <nano/lsearchk/cgdescent.h>
#include <nano/lsearchk/fletcher.h>
#include <nano/lsearchk/lemarechal.h>
#include <nano/lsearchk/morethuente.h>

using namespace nano;

static auto make_functions()
{
    return function_t::make({1, 16, convexity::ignore, smoothness::yes, 10}, std::regex(".+"));
}

static auto get_lsearch(const string_t& id)
{
    auto lsearch = lsearchk_t::all().get(id);
    UTEST_REQUIRE(lsearch);
    return lsearch;
}

static void setup_logger(lsearchk_t& lsearch, std::stringstream& stream)
{
    const auto [c1, c2] = lsearch.parameter("lsearchk::tolerance").value_pair<scalar_t>();

    // log the line-search trials
    lsearch.logger(
        [&, c1 = c1, c2 = c2](const solver_state_t& state0, const solver_state_t& state)
        {
            stream << "\tt=" << state.t << ",f=" << state.f << ",g=" << state.convergence_criterion()
                   << ",armijo=" << state.has_armijo(state0, c1) << ",wolfe=" << state.has_wolfe(state0, c2)
                   << ",swolfe=" << state.has_strong_wolfe(state0, c2)
                   << ",awolfe=" << state.has_approx_wolfe(state0, c1, c2) << ".\n";
        });
}

static void test(const rlsearchk_t& lsearch, const function_t& function, const vector_t& x0, const scalar_t t0,
                 const std::tuple<scalar_t, scalar_t>& c12)
{
    UTEST_REQUIRE_NOTHROW(lsearch->parameter("lsearchk::tolerance") = c12);

    const auto [c1, c2]       = c12;
    const auto old_n_failures = utest_n_failures.load();

    auto state0 = solver_state_t{function, x0};
    UTEST_CHECK(state0.valid());
    state0.d = -state0.g;
    UTEST_CHECK(state0.has_descent());

    std::stringstream stream;
    stream << std::fixed << std::setprecision(12) << function.name() << " " << lsearch->type_id() << ": x0=["
           << state0.x.transpose() << "],t0=" << t0 << ",f0=" << state0.f << ",g0=" << state0.convergence_criterion()
           << "\n";
    setup_logger(*lsearch, stream);

    const auto cgdescent_epsilon = [&]()
    { return lsearch->parameter("lsearchk::cgdescent::epsilon").value<scalar_t>(); };

    // check the Armijo and the Wolfe-like conditions are valid after line-search
    auto state = state0;
    UTEST_CHECK(lsearch->get(state, t0));
    UTEST_CHECK(state.valid());
    UTEST_CHECK_GREATER(state.t, 0.0);
    UTEST_CHECK_LESS_EQUAL(state.f, state0.f);

    switch (lsearch->type())
    {
    case lsearch_type::armijo: UTEST_CHECK(state.has_armijo(state0, c1)); break;

    case lsearch_type::wolfe:
        UTEST_CHECK(state.has_armijo(state0, c1));
        UTEST_CHECK(state.has_wolfe(state0, c2));
        break;

    case lsearch_type::strong_wolfe:
        UTEST_CHECK(state.has_armijo(state0, c1));
        UTEST_CHECK(state.has_strong_wolfe(state0, c2));
        break;

    case lsearch_type::wolfe_approx_wolfe:
        UTEST_CHECK((state.has_armijo(state0, c1) && state.has_wolfe(state0, c2)) ||
                    (state.has_approx_armijo(state0, cgdescent_epsilon() * std::fabs(state0.f)) &&
                     state.has_approx_wolfe(state0, c1, c2)));
        break;

    default: break;
    }

    if (old_n_failures != utest_n_failures.load())
    {
        std::cout << stream.str();
    }
}

static void test(const rlsearchk_t& lsearch, const function_t& function)
{
    for (const auto& x0 : make_random_x0s(function))
    {
        for (const auto& c12 : std::vector<std::tuple<scalar_t, scalar_t>>{
                 {1e-4, 1e-1},
                 {1e-4, 9e-1},
                 {1e-1, 9e-1}
        })
        {
            test(lsearch, function, x0, 1e-1, c12);
            test(lsearch, function, x0, 3e-1, c12);
            test(lsearch, function, x0, 1e+0, c12);
            test(lsearch, function, x0, 3e+1, c12);
        }
    }
}

UTEST_BEGIN_MODULE(test_lsearch)

UTEST_CASE(interpolate)
{
    const auto u = lsearch_step_t{4.2, 1.0, 0.5};
    const auto v = u;

    const auto tc = lsearch_step_t::interpolate(u, v, interpolation_type::cubic);
    const auto tq = lsearch_step_t::interpolate(u, v, interpolation_type::quadratic);
    const auto tb = lsearch_step_t::interpolate(u, v, interpolation_type::bisection);

    UTEST_CHECK_CLOSE(tc, 4.2, 1e-16);
    UTEST_CHECK_CLOSE(tq, 4.2, 1e-16);
    UTEST_CHECK_CLOSE(tb, 4.2, 1e-16);
}

UTEST_CASE(backtrack_cubic)
{
    const auto lsearch                                       = get_lsearch("backtrack");
    lsearch->parameter("lsearchk::backtrack::interpolation") = interpolation_type::cubic;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(backtrack_quadratic)
{
    const auto lsearch                                       = get_lsearch("backtrack");
    lsearch->parameter("lsearchk::backtrack::interpolation") = interpolation_type::quadratic;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(backtrack_bisection)
{
    const auto lsearch                                       = get_lsearch("backtrack");
    lsearch->parameter("lsearchk::backtrack::interpolation") = interpolation_type::bisection;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(lemarechal_cubic)
{
    const auto lsearch                                        = get_lsearch("lemarechal");
    lsearch->parameter("lsearchk::lemarechal::interpolation") = interpolation_type::cubic;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(lemarechal_quadratic)
{
    const auto lsearch                                        = get_lsearch("lemarechal");
    lsearch->parameter("lsearchk::lemarechal::interpolation") = interpolation_type::quadratic;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(lemarechal_bisection)
{
    const auto lsearch                                        = get_lsearch("lemarechal");
    lsearch->parameter("lsearchk::lemarechal::interpolation") = interpolation_type::bisection;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(morethuente)
{
    const auto lsearch = get_lsearch("morethuente");

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(fletcher_cubic)
{
    const auto lsearch                                      = get_lsearch("fletcher");
    lsearch->parameter("lsearchk::fletcher::interpolation") = interpolation_type::cubic;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(fletcher_quadratic)
{
    const auto lsearch                                      = get_lsearch("fletcher");
    lsearch->parameter("lsearchk::fletcher::interpolation") = interpolation_type::quadratic;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(fletcher_bisection)
{
    const auto lsearch                                      = get_lsearch("fletcher");
    lsearch->parameter("lsearchk::fletcher::interpolation") = interpolation_type::bisection;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(cgdescent)
{
    const auto lsearch = get_lsearch("cgdescent");

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_END_MODULE()
