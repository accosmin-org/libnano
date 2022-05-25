#include <iomanip>
#include <nano/solver.h>
#include <nano/tensor.h>
#include <nano/core/tpool.h>
#include <nano/core/chrono.h>
#include <nano/core/logger.h>
#include <nano/core/cmdline.h>
#include <nano/core/numeric.h>
#include <nano/core/factory_util.h>
#include <nano/function/benchmark.h>

using namespace nano;

struct result_t
{
    result_t() = default;
    result_t(const solver_state_t& state, int64_t milliseconds) :
        m_value(state.f),
        m_gnorm(state.convergence_criterion()),
        m_status(state.m_status),
        m_fcalls(state.m_fcalls),
        m_gcalls(state.m_gcalls),
        m_milliseconds(milliseconds)
    {
    }

    scalar_t                m_value{0.0};
    scalar_t                m_gnorm{0.0};
    solver_state_t::status  m_status{solver_state_t::status::converged};
    tensor_size_t           m_fcalls{0};
    tensor_size_t           m_gcalls{0};
    int64_t                 m_milliseconds{0};
};

struct solver_function_stats_t
{
    explicit solver_function_stats_t(size_t trials) :
        m_values(static_cast<tensor_size_t>(trials)),
        m_gnorms(static_cast<tensor_size_t>(trials)),
        m_errors(static_cast<tensor_size_t>(trials)),
        m_maxits(static_cast<tensor_size_t>(trials)),
        m_fcalls(static_cast<tensor_size_t>(trials)),
        m_gcalls(static_cast<tensor_size_t>(trials)),
        m_millis(static_cast<tensor_size_t>(trials)),
        m_ranks(static_cast<tensor_size_t>(trials)),
        m_precisions(static_cast<tensor_size_t>(trials))
    {
    }

    void update(size_t trial, const result_t& result, scalar_t precision, ptrdiff_t rank)
    {
        const auto itrial = static_cast<tensor_size_t>(trial);
        m_values(itrial) = result.m_value;
        m_gnorms(itrial) = result.m_gnorm;
        m_errors(itrial) = result.m_status == solver_state_t::status::failed ? 1.0 : 0.0;
        m_maxits(itrial) = result.m_status == solver_state_t::status::max_iters ? 1.0 : 0.0;
        m_fcalls(itrial) = static_cast<scalar_t>(result.m_fcalls);
        m_gcalls(itrial) = static_cast<scalar_t>(result.m_gcalls);
        m_millis(itrial) = static_cast<scalar_t>(result.m_milliseconds);
        m_ranks(itrial) = static_cast<scalar_t>(rank);
        m_precisions(itrial) = precision;
    }

    // attributes
    tensor1d_t              m_values;       ///< function values
    tensor1d_t              m_gnorms;       ///< gradient norms
    tensor1d_t              m_errors;       ///< #internal errors (e.g. line-search failed)
    tensor1d_t              m_maxits;       ///< #maximum iterations reached (without convergence)
    tensor1d_t              m_fcalls;       ///< #function value calls
    tensor1d_t              m_gcalls;       ///< #gradient calls
    tensor1d_t              m_millis;       ///< number of milliseconds
    tensor1d_t              m_ranks;        ///< rank as ordered by the function value
    tensor1d_t              m_precisions;   ///< relative precision to the best solver
};

struct solver_stats_t
{
    explicit solver_stats_t(size_t functions) :
        m_ranks(static_cast<tensor_size_t>(functions)),
        m_precisions(static_cast<tensor_size_t>(functions))
    {
    }

    void update(size_t function, scalar_t precision, scalar_t rank)
    {
        const auto ifunction = static_cast<tensor_size_t>(function);
        m_ranks(ifunction) = static_cast<scalar_t>(rank);
        m_precisions(ifunction) = precision;
    }

    // attributes
    tensor1d_t              m_ranks;        ///< rank as ordered by the function value
    tensor1d_t              m_precisions;   ///< relative precision to the best solver
};

static auto relative_precision(scalar_t value, scalar_t best_value)
{
    assert(value >= best_value);
    return std::log10(std::max(value - best_value, std::numeric_limits<scalar_t>::epsilon()));
}

static auto relative_precision(const result_t& result, const result_t& best_result)
{
    return relative_precision(result.m_value, best_result.m_value);
}

static auto make_solver_name(const std::pair<string_t, rsolver_t>& solver_info)
{
    const auto& solver = solver_info.second;
    const auto& solver_id = solver_info.first;

    return solver->monotonic() ?
        scat(solver_id, " [", solver->lsearch0_id(), ",", solver->lsearchk_id(), "]") :
        solver_id;
}

using points_t = std::vector<vector_t>;
using results_t = std::vector<result_t>;
using solvers_t = std::vector<std::pair<string_t, rsolver_t>>;

static auto log_solver(const function_t& function, const rsolver_t& solver, const string_t& solver_id,
    const vector_t& x0)
{
    std::cout << std::fixed << std::setprecision(10);
    std::cout << function.name()
        << " solver[" << solver_id
        << "],lsearch0[" << (solver->monotonic() ? solver->lsearch0_id() : string_t("N/A"))
        << "],lsearchk[" << (solver->monotonic() ? solver->lsearchk_id() : string_t("N/A"))
        << "]" << std::endl;

    solver->logger([&] (const solver_state_t& state)
    {
        std::cout
            << "descent: i=" << state.m_iterations << ",f=" << state.f << ",g=" << state.convergence_criterion()
            << "[" << state.m_status << "]" << ",calls=" << state.m_fcalls << "/" << state.m_gcalls << "." << std::endl;
        return true;
    });

    solver->lsearch0_logger([&] (const solver_state_t& state0, const scalar_t t)
    {
        std::cout
            << "\tlsearch(0): t=" << state0.t << ",f=" << state0.f << ",g=" << state0.convergence_criterion()
            << ",t=" << t << "." << std::endl;
    });

    const auto [c1, c2] = solver->parameter("solver::tolerance").value_pair<scalar_t>();

    solver->lsearchk_logger([&, c1=c1, c2=c2] (const solver_state_t& state0, const solver_state_t& state)
    {
        std::cout
            << "\tlsearch(t): t=" << state.t << ",f=" << state.f << ",g=" << state.convergence_criterion()
            << ",armijo=" << state.has_armijo(state0, c1)
            << ",wolfe=" << state.has_wolfe(state0, c2)
            << ",swolfe=" << state.has_strong_wolfe(state0, c2) << "." << std::endl;
    });

    auto state = solver->minimize(function, x0);
    std::cout << std::flush;

    // NB: need to reset the logger for the next batch of tests!
    solver->logger({});
    solver->lsearch0_logger({});
    solver->lsearchk_logger({});

    return state;
}

static auto minimize_all(const function_t& function, const solvers_t& solvers,
    const points_t& x0s, bool log_failures, bool log_maxits)
{
    results_t results{x0s.size() * solvers.size()};
    loopi(results.size(), [&] (size_t i, size_t)
    {
        const auto timer = nano::timer_t{};

        const auto& x0 = x0s[i / solvers.size()];
        const auto& solver = solvers[i % solvers.size()].second;
        const auto state = solver->minimize(function, x0);

        const auto milliseconds = timer.milliseconds().count();

        results[i] = result_t{state, milliseconds};
    });

    for (size_t i = 0; i < results.size() && (log_failures || log_maxits); ++ i)
    {
        // log in full detail the optimization trajectory if it fails
        if ((results[i].m_status == solver_state_t::status::max_iters && log_maxits) ||
            (results[i].m_status == solver_state_t::status::failed && log_failures))
        {
            const auto& x0 = x0s[i / solvers.size()];
            const auto& solver = solvers[i % solvers.size()].second;
            const auto& solver_id = solvers[i % solvers.size()].first;
            const auto state = log_solver(function, solver, solver_id, x0);
            assert(state.m_status == results[i].m_status);
        }
    }

    return results;
}

static auto benchmark(const function_t& function, const solvers_t& solvers,
    size_t trials, bool log_failures, bool log_maxits)
{
    // generate a fixed set of random initial points
    points_t x0s(trials);
    std::generate(x0s.begin(), x0s.end(), [&] () { return vector_t::Random(function.size()); });

    // and minimize in parallel all (solver, random initial point) combinations
    const auto results = minimize_all(function, solvers, x0s, log_failures, log_maxits);

    // gather statistics per solver
    const auto max_evals = solvers[0U].second->parameter("solver::max_evals").value<int>();
    const auto max_digits_calls = static_cast<size_t>(std::log10(max_evals)) + 1U;

    auto stats = std::vector<solver_function_stats_t>{solvers.size(), solver_function_stats_t{trials}};
    auto ranks = std::vector<std::pair<scalar_t, size_t>>{solvers.size()};

    for (size_t trial = 0U; trial < trials; ++ trial)
    {
        const auto* const begin = &results[trial * solvers.size()];
        const auto* const end = &results[trial * solvers.size() + solvers.size()];

        const auto& best_result = *std::min_element(
            begin, end, [] (const auto& lhs, const auto& rhs) { return lhs.m_value < rhs.m_value; });

        for (size_t isolver = 0U; isolver < solvers.size(); ++ isolver)
        {
            const auto& result = results[trial * solvers.size() + isolver];
            ranks[isolver] = std::make_pair(result.m_value, isolver);
        }
        std::sort(ranks.begin(), ranks.end());

        for (size_t isolver = 0U; isolver < solvers.size(); ++ isolver)
        {
            const auto& result = results[trial * solvers.size() + isolver];
            assert(std::isfinite(result.m_value));
            assert(std::isfinite(result.m_gnorm));

            const auto precision = relative_precision(result, best_result);

            const auto find = [&] (const auto& v) { return v.second == isolver; };
            const auto rank = (std::find_if(ranks.begin(), ranks.end(), find) - ranks.begin()) + 1;

            stats[isolver].update(trial, result, precision, rank);
        }
    }

    // display per-function statistics
    table_t table;
    table.header()
        << align(function.name(), 32)
        << align("precision", 9) << align("rank", 4) << align("value", 12) << align("gnorm", 12)
        << "errors" << "maxits"
        << align("fcalls", max_digits_calls)
        << align("gcalls", max_digits_calls) << align("[ms]", 5);
    table.delim();

    for (size_t isolver = 0U; isolver < solvers.size(); ++ isolver)
    {
        const auto& stat = stats[isolver];
        const auto solver_name = make_solver_name(solvers[isolver]);

        table.append()
            << solver_name
            << scat(std::fixed, std::setprecision(4), stat.m_precisions.mean())
            << scat(std::fixed, std::setprecision(2), stat.m_ranks.mean())
            << stat.m_values.mean()
            << stat.m_gnorms.mean()
            << static_cast<size_t>(stat.m_errors.sum())
            << static_cast<size_t>(stat.m_maxits.sum())
            << static_cast<size_t>(stat.m_fcalls.mean())
            << static_cast<size_t>(stat.m_gcalls.mean())
            << static_cast<size_t>(stat.m_millis.mean());
    }

    table.sort(nano::make_less_from_string<scalar_t>(), {2}); // NB: sort solvers by precision!
    std::cout << table;

    return stats;
}

static int unsafe_main(int argc, const char* argv[])
{
    using namespace nano;

    // parse the command line
    cmdline_t cmdline("benchmark solvers");
    cmdline.add("", "solver",           "regex to select the line-search solvers to benchmark", ".+");
    cmdline.add("", "function",         "regex to select the functions to benchmark", ".+");
    cmdline.add("", "min-dims",         "minimum number of dimensions for each test function (if feasible)", "4");
    cmdline.add("", "max-dims",         "maximum number of dimensions for each test function (if feasible)", "16");
    cmdline.add("", "trials",           "number of random trials for each test function", "100");
    cmdline.add("", "convex",           "use only convex test functions");
    cmdline.add("", "smooth",           "use only smooth test functions");
    cmdline.add("", "non-smooth",       "use only non-smooth test functions");
    cmdline.add("", "lsearch0",         "use this regex to select the line-search initialization methods");
    cmdline.add("", "lsearchk",         "use this regex to select the line-search strategies");
    cmdline.add("", "log-failures",     "log the optimization trajectory for the runs that fail");
    cmdline.add("", "log-maxits",       "log the optimization trajectory that failed to converge");
    cmdline.add("", "list-solver",      "list the available solvers");
    cmdline.add("", "list-function",    "list the available test functions");
    cmdline.add("", "list-lsearch0",    "list the available line-search initialization methods");
    cmdline.add("", "list-lsearchk",    "list the available line-search strategies");
    cmdline.add("", "list-solver-params","list the available parameters for each of the selected solvers");

    const auto options = cmdline.process(argc, argv);

    if (options.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    if (options.has("list-solver"))
    {
        std::cout << make_table("solver", solver_t::all());
        return EXIT_SUCCESS;
    }

    if (options.has("list-function"))
    {
        std::cout << make_table("function", benchmark_function_t::all());
        return EXIT_SUCCESS;
    }

    if (options.has("list-lsearch0"))
    {
        std::cout << make_table("lsearch0", lsearch0_t::all());
        return EXIT_SUCCESS;
    }

    if (options.has("list-lsearchk"))
    {
        std::cout << make_table("lsearchk", lsearchk_t::all());
        return EXIT_SUCCESS;
    }

    if (options.has("list-solver-params"))
    {
        const auto sregex = std::regex(options.get<string_t>("solver"));
        const auto solver_ids = solver_t::all().ids(sregex);

        table_t table;
        table.header() << "solver" << "parameter" << "value" << "domain";
        table.delim();
        for (const auto& solver_id : solver_ids)
        {
            const auto solver = solver_t::all().get(solver_id);
            for (const auto& param : solver->parameters())
            {
                table.append() << solver_id << param.name() << param.value() << param.domain();
            }
            if (&solver_id != &*solver_ids.rbegin())
            {
                table.delim();
            }
        }

        std::cout << table;
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto min_dims = options.get<tensor_size_t>("min-dims");
    const auto max_dims = options.get<tensor_size_t>("max-dims");
    const auto trials = options.get<size_t>("trials");
    const auto convex = options.has("convex") ? convexity::yes : convexity::ignore;
    const auto smooth = options.has("smooth") ? smoothness::yes : (options.has("non-smooth") ? smoothness::no : smoothness::ignore);
    const auto log_failures = options.has("log-failures");
    const auto log_maxits = options.has("log-maxits");

    const auto fregex = std::regex(options.get<string_t>("function"));
    const auto sregex = std::regex(options.get<string_t>("solver"));

    const auto lsearch0s = options.has("lsearch0") ?
        lsearch0_t::all().ids(std::regex(options.get<string_t>("lsearch0"))) :
        strings_t{""};

    const auto lsearchks = options.has("lsearchk") ?
        lsearchk_t::all().ids(std::regex(options.get<string_t>("lsearchk"))) :
        strings_t{""};

    const auto solver_ids = solver_t::all().ids(sregex);
    critical(
        solver_ids.empty(),
        "at least a solver needs to be selected!");

    const auto functions = benchmark_function_t::make({min_dims, max_dims, convex, smooth}, fregex);
    critical(
        functions.empty(),
        "at least a function needs to be selected!");

    // keep track of the used parameters
    std::map<string_t, int> params_usage;
    for (const auto& [param_name, param_value] : options.m_xvalues)
    {
        params_usage[param_name] = 0;
    }

    // construct the list of solver configurations to evaluate
    solvers_t solvers;
    const auto add_solver = [&] (const string_t& solver_id, rsolver_t&& solver)
    {
        // setup solver with additional parameters
        for (const auto& [param_name, param_value] : options.m_xvalues)
        {
            if (solver->parameter_if(param_name) != nullptr)
            {
                solver->parameter(param_name) = param_value;
                params_usage[param_name] ++;
            }
        }

        solvers.emplace_back(solver_id, std::move(solver));
    };

    for (const auto& solver_id : solver_ids)
    {
        auto solver = solver_t::all().get(solver_id);
        if (solver->monotonic())
        {
            for (const auto& lsearch0 : lsearch0s)
            {
                for (const auto& lsearchk : lsearchks)
                {
                    solver = solver_t::all().get(solver_id);
                    if (!lsearch0.empty()) { solver->lsearch0(lsearch0); }
                    if (!lsearchk.empty()) { solver->lsearchk(lsearchk); }
                    add_solver(solver_id, std::move(solver));
                }
            }
        }
        else
        {
            add_solver(solver_id, std::move(solver));
        }
    }

    // benchmark solvers independently per function
    auto solver_stats = std::vector<solver_stats_t>{solvers.size(), solver_stats_t{functions.size()}};
    for (size_t ifunction = 0U; ifunction < functions.size(); ++ ifunction)
    {
        const auto& function = functions[ifunction];
        const auto solver_function_stats = benchmark(*function, solvers, trials, log_failures, log_maxits);
        for (size_t isolver = 0U; isolver < solvers.size(); ++ isolver)
        {
            const auto& stats = solver_function_stats[isolver];
            solver_stats[isolver].update(ifunction, stats.m_precisions.mean(), stats.m_ranks.mean());
        }
    }

    // display global statistics
    table_t table;
    table.header() << align("solver", 32) << align("precision", 9) << align("rank", 4);
    table.delim();

    for (size_t isolver = 0U; isolver < solvers.size(); ++ isolver)
    {
        const auto& stat = solver_stats[isolver];
        const auto solver_name = make_solver_name(solvers[isolver]);

        table.append()
            << solver_name
            << scat(std::fixed, std::setprecision(4), stat.m_precisions.mean())
            << scat(std::fixed, std::setprecision(2), stat.m_ranks.mean());
    }

    table.sort(nano::make_less_from_string<scalar_t>(), {2}); // NB: sort solvers by precision!
    std::cout << table;

    // log all unused parameters (e.g. typos, not matching to any solver)
    for (const auto& [param_name, count] : params_usage)
    {
        if (count == 0)
        {
            log_warning() << "parameter \"" << param_name << "\" was not used.";
        }
    }

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
