#include <nano/core/numeric.h>
#include <nano/solver.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] inline auto make_solver(const string_t& name = "cgd-n", const scalar_t epsilon = 1e-8,
                                         const int max_evals = 20000)
{
    auto solver = solver_t::all().get(name);
    UTEST_REQUIRE(solver);
    solver->parameter("solver::epsilon")   = epsilon;
    solver->parameter("solver::max_evals") = max_evals;
    return solver;
}

static void setup_logger(solver_t& solver, std::stringstream& stream)
{
    solver.logger(
        [&](const solver_state_t& state)
        {
            stream << "\tdescent: " << state << ",x=" << state.x().transpose() << ".\n ";
            return true;
        });

    solver.lsearch0_logger(
        [&](const solver_state_t& state, const scalar_t step_size) {
            stream << "\t\tlsearch(0): t=" << step_size << ",f=" << state.fx() << ",g=" << state.gradient_test()
                   << ".\n";
        });

    const auto [c1, c2] = solver.parameter("solver::tolerance").value_pair<scalar_t>();

    solver.lsearchk_logger(
        [&, c1 = c1, c2 = c2](const solver_state_t& state0, const solver_state_t& state, const vector_t& descent,
                              const scalar_t step_size)
        {
            stream << "\t\tlsearch(t): t=" << step_size << ",f=" << state.fx() << ",g=" << state.gradient_test()
                   << ",armijo=" << state.has_armijo(state0, descent, step_size, c1)
                   << ",wolfe=" << state.has_wolfe(state0, descent, c2)
                   << ",swolfe=" << state.has_strong_wolfe(state0, descent, c2) << ".\n";
        });
}

struct minimize_config_t
{
    auto& config(const minimize_config_t& value)
    {
        m_epsilon                    = value.m_epsilon;
        m_max_evals                  = value.m_max_evals;
        m_expected_convergence       = value.m_expected_convergence;
        m_expected_maximum_deviation = value.m_expected_maximum_deviation;
        return *this;
    }

    auto& epsilon(const scalar_t value)
    {
        m_epsilon = value;
        return *this;
    }

    auto& max_evals(const tensor_size_t value)
    {
        m_max_evals = value;
        return *this;
    }

    auto& expected_convergence(const bool value)
    {
        m_expected_convergence = value;
        return *this;
    }

    auto& expected_minimum(const scalar_t value)
    {
        if (!std::isfinite(m_expected_minimum))
        {
            m_expected_minimum = value;
        }
        return *this;
    }

    auto& expected_maximum_deviation(const scalar_t value)
    {
        m_expected_maximum_deviation = value;
        return *this;
    }

    scalar_t      m_epsilon{1e-6};
    tensor_size_t m_max_evals{50000};
    bool          m_expected_convergence{true};
    scalar_t      m_expected_minimum{std::numeric_limits<scalar_t>::quiet_NaN()};
    scalar_t      m_expected_maximum_deviation{1e-6};
};

struct solver_description_t
{
    solver_description_t() = default;

    explicit solver_description_t(const solver_type type)
        : m_type(type)
    {
    }

    auto& smooth_config(const minimize_config_t& config)
    {
        m_smooth_config = config;
        return *this;
    }

    auto& nonsmooth_config(const minimize_config_t& config)
    {
        m_nonsmooth_config = config;
        return *this;
    }

    solver_type       m_type{solver_type::line_search};
    minimize_config_t m_smooth_config{};
    minimize_config_t m_nonsmooth_config{};
};

[[maybe_unused]] static auto make_description(const string_t& solver_id)
{
    if (solver_id == "cgd-n" || solver_id == "cgd-hs" || solver_id == "cgd-fr" || solver_id == "cgd-pr" ||
        solver_id == "cgd-cd" || solver_id == "cgd-ls" || solver_id == "cgd-dy" || solver_id == "cgd-dycd" ||
        solver_id == "cgd-dyhs" || solver_id == "cgd-frpr" || solver_id == "lbfgs" || solver_id == "sr1" ||
        solver_id == "bfgs" || solver_id == "hoshino" || solver_id == "fletcher")
    {
        return solver_description_t{solver_type::line_search}.smooth_config(
            minimize_config_t{}.epsilon(5e-8).max_evals(1000).expected_maximum_deviation(1e-6));
    }
    else if (solver_id == "dfp")
    {
        // NB: DFP needs many more iterations to reach the solution!
        return solver_description_t{solver_type::line_search}.smooth_config(
            minimize_config_t{}.epsilon(5e-8).max_evals(20000).expected_maximum_deviation(1e-6));
    }
    else if (solver_id == "gd")
    {
        // NB: gradient descent (GD) needs many more iterations to minimize badly conditioned problems!
        return solver_description_t{solver_type::line_search}.smooth_config(
            minimize_config_t{}.epsilon(5e-7).max_evals(10000).expected_maximum_deviation(1e-5));
    }
    else if (solver_id == "ellipsoid" || solver_id == "osga")
    {
        // NB: very precise for both smooth and non-smooth problems.
        // NB: the ellipsoid method is reasonably fast only for very low-dimensional problems.
        // NB: the stopping criterion is working very well in practice.
        return solver_description_t{solver_type::non_monotonic}
            .smooth_config(minimize_config_t{}.epsilon(5e-8).expected_maximum_deviation(1e-6))
            .nonsmooth_config(minimize_config_t{}.epsilon(5e-8).expected_maximum_deviation(1e-5));
    }
    else if (solver_id == "sgm" || solver_id == "cocob" || solver_id == "sda" ||
             solver_id == "wda" ||                                           // primal-dual subgradient method
             solver_id == "pgm" || solver_id == "dgm" || solver_id == "fgm") // universal gradient methods
    {
        // NB: unreliable methods:
        // - either no theoretical or practical stopping criterion
        // - very slow convergence rate for both non-smooth and hard smooth problems
        return solver_description_t{solver_type::non_monotonic}
            .smooth_config(
                minimize_config_t{}.max_evals(1000).expected_convergence(false).expected_maximum_deviation(1e+1))
            .nonsmooth_config(
                minimize_config_t{}.max_evals(1000).expected_convergence(false).expected_maximum_deviation(1e+1));
    }
    else
    {
        assert(false);
        return solver_description_t{};
    }
}

[[maybe_unused]] static auto check_minimize(solver_t& solver, const function_t& function, const vector_t& x0,
                                            const minimize_config_t& config = minimize_config_t{})
{
    const auto old_n_failures = utest_n_failures.load();
    const auto state0         = solver_state_t{function, x0};

    const auto solver_id   = solver.type_id();
    const auto lsearch0_id = solver.type() == solver_type::line_search ? solver.lsearch0().type_id() : "N/A";
    const auto lsearchk_id = solver.type() == solver_type::line_search ? solver.lsearchk().type_id() : "N/A";

    std::stringstream stream;
    stream << std::fixed << std::setprecision(19) << function.name() << " " << solver_id << "[" << lsearch0_id << ","
           << lsearchk_id << "]\n"
           << ":x0=[" << state0.x().transpose() << "],f0=" << state0.fx() << ",g0=" << state0.gradient_test();
    if (state0.ceq().size() + state0.cineq().size() > 0)
    {
        stream << ",c0=" << state0.constraint_test() << "\n";
    }

    setup_logger(solver, stream);

    // minimize
    solver.parameter("solver::epsilon")   = config.m_epsilon;
    solver.parameter("solver::max_evals") = config.m_max_evals;

    function.clear_statistics();
    auto state = solver.minimize(function, x0);

    UTEST_CHECK(state.valid());
    UTEST_CHECK_LESS_EQUAL(state.fx(), state0.fx() + epsilon1<scalar_t>());
    if (function.smooth() && solver.type() == solver_type::line_search)
    {
        UTEST_CHECK_LESS(state.gradient_test(), config.m_epsilon);
    }
    if (config.m_expected_convergence)
    {
        UTEST_CHECK_EQUAL(state.status(), solver_status::converged);
    }
    else
    {
        UTEST_CHECK_NOT_EQUAL(state.status(), solver_status::failed);
    }
    UTEST_CHECK_EQUAL(state.fcalls(), function.fcalls());
    UTEST_CHECK_EQUAL(state.gcalls(), function.gcalls());
    if (function.convex() && std::isfinite(config.m_expected_minimum))
    {
        UTEST_CHECK_CLOSE(state.fx(), config.m_expected_minimum, config.m_expected_maximum_deviation);
    }

    if (old_n_failures != utest_n_failures.load())
    {
        std::cout << stream.str();
    }

    return state;
}
