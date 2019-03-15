#include <iostream>
#include <nano/solver.h>
#include <nano/stats.h>
#include <nano/table.h>
#include <nano/tpool.h>
#include <nano/logger.h>
#include <nano/cmdline.h>
#include <nano/numeric.h>

using namespace nano;

struct solver_stat_t
{
    void update(const solver_state_t& state)
    {
        m_crits(state.convergence_criterion());
        m_fails(state.m_status != solver_state_t::status::converged ? 1 : 0);
        m_iters(static_cast<scalar_t>(state.m_iterations));
        m_errors(state.m_status == solver_state_t::status::failed ? 1 : 0);
        m_maxits(state.m_status == solver_state_t::status::max_iters ? 1 : 0);
        m_fcalls(static_cast<scalar_t>(state.m_fcalls));
        m_gcalls(static_cast<scalar_t>(state.m_gcalls));
        m_costs(static_cast<scalar_t>(state.m_fcalls + 2 * state.m_gcalls));
    }

    stats_t     m_crits;    ///< convergence criterion
    stats_t     m_fails;    ///< #convergence failures
    stats_t     m_iters;    ///< #optimization iterations
    stats_t     m_errors;   ///< #internal errors (e.g. line-search failed)
    stats_t     m_maxits;   ///< #maximum iterations reached
    stats_t     m_fcalls;   ///< #function value calls
    stats_t     m_gcalls;   ///< #gradient calls
    stats_t     m_costs;    ///< computation cost as a function of function value and gradient calls
};

using solver_config_stats_t = std::map<
    std::pair<string_t, string_t>,  ///< key: {solver id, solver config}
    solver_stat_t>;                 ///< value: solver statistics

static void show_table(const string_t& table_name, const solver_config_stats_t& stats)
{
    assert(!stats.empty());

    // show global statistics
    table_t table;
    table.header()
        << colspan(2) << table_name
        << "gnorm"
        << "#fails"
        << "#iters"
        << "#errors"
        << "#maxits"
        << "#fcalls"
        << "#gcalls"
        << "cost";
    table.delim();

    for (const auto& it : stats)
    {
        const auto& id = it.first.first;
        const auto& config = it.first.second;
        const auto& stat = it.second;

        if (stat.m_fcalls)
        {
            table.append()
            << id
            << config
            << stat.m_crits.avg()
            << static_cast<size_t>(stat.m_fails.sum1())
            << static_cast<size_t>(stat.m_iters.avg())
            << static_cast<size_t>(stat.m_errors.sum1())
            << static_cast<size_t>(stat.m_maxits.sum1())
            << static_cast<size_t>(stat.m_fcalls.avg())
            << static_cast<size_t>(stat.m_gcalls.avg())
            << static_cast<size_t>(stat.m_costs.avg());
        }
    }

    table.sort(nano::make_less_from_string<scalar_t>(), {3, 9});
    std::cout << table;
}

static auto trim(const json_t& json)
{
    string_t config = json.dump();
    config = nano::replace(config, ",,", ",");
    config = nano::replace(config, "\"", "");
    config = nano::replace(config, ",}", "");
    config = nano::replace(config, "}", "");
    config = nano::replace(config, "{", "");

    return config;
}

static void check_solver(const function_t& function, const rsolver_t& solver, const string_t& id,
    const std::vector<vector_t>& x0s,
    solver_config_stats_t& fstats, solver_config_stats_t& gstats)
{
    const auto json = solver->config();
    const auto config = trim(json);

    std::vector<solver_state_t> states(x0s.size());
    nano::loopi(x0s.size(), [&] (const size_t i)
    {
        states[i] = solver->minimize(function, x0s[i]);
    });

    for (const auto& state : states)
    {
        fstats[std::make_pair(id, config)].update(state);
        gstats[std::make_pair(id, config)].update(state);
    }
}

static void check_function(const function_t& function, const std::vector<std::pair<string_t, rsolver_t>>& id_solvers,
    const size_t trials, solver_config_stats_t& gstats)
{
    // generate fixed random trials
    std::vector<vector_t> x0s(trials);
    std::generate(x0s.begin(), x0s.end(), [&] () { return vector_t::Random(function.size()); });

    // per-problem statistics
    solver_config_stats_t fstats;

    // evaluate all possible combinations (solver & line-search)
    for (const auto& id_solver : id_solvers)
    {
        const auto& id = id_solver.first;
        const auto& solver = id_solver.second;

        check_solver(function, solver, id, x0s, fstats, gstats);
    }

    // show per-problem statistics
    show_table(function.name(), fstats);
}

static int unsafe_main(int argc, const char* argv[])
{
    using namespace nano;

    // parse the command line
    cmdline_t cmdline("benchmark solvers");
    cmdline.add("", "solvers",      "use this regex to select the solvers to benchmark", ".+");
    cmdline.add("", "functions",    "use this regex to select the functions to benchmark", ".+");
    cmdline.add("", "min-dims",     "minimum number of dimensions for each test function (if feasible)", "100");
    cmdline.add("", "max-dims",     "maximum number of dimensions for each test function (if feasible)", "1000");
    cmdline.add("", "trials",       "number of random trials for each test function", "100");
    cmdline.add("", "iterations",   "maximum number of iterations", "1000");
    cmdline.add("", "epsilon",      "convergence criterion", epsilon2<scalar_t>());
    cmdline.add("", "convex",       "use only convex test functions");
    cmdline.add("", "c1",           "use this c1 value (see Armijo-Goldstein line-search step condition)");
    cmdline.add("", "c2",           "use this c2 value (see Wolfe line-search step condition)");
    cmdline.add("", "ls-init",      "use this regex to select the line-search initialization methods");
    cmdline.add("", "ls-strategy",  "use this regex to select the line-search methods");

    cmdline.process(argc, argv);

    // check arguments and options
    const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
    const auto max_dims = cmdline.get<tensor_size_t>("max-dims");
    const auto trials = cmdline.get<size_t>("trials");
    const auto iterations = cmdline.get<size_t>("iterations");
    const auto epsilon = cmdline.get<scalar_t>("epsilon");
    const auto is_convex = cmdline.has("convex");

    const auto fregex = std::regex(cmdline.get<string_t>("functions"));
    const auto sregex = std::regex(cmdline.get<string_t>("solvers"));

    const auto ls_inits = cmdline.has("ls-init") ?
        lsearch_init_t::all().ids(std::regex(cmdline.get<string_t>("ls-init"))) :
        strings_t{};

    const auto ls_strategies = cmdline.has("ls-strategy") ?
        lsearch_strategy_t::all().ids(std::regex(cmdline.get<string_t>("ls-strategy"))) :
        strings_t{};

    // construct the list of solver configurations to evaluate
    std::vector<std::pair<string_t, rsolver_t>> solvers;
    const auto add_solver = [&] (const auto& solver_id, const auto* ls_init, const auto* ls_strategy)
    {
        auto solver = solver_t::all().get(solver_id);
        if (cmdline.has("c1"))
        {
            solver->config(nano::to_json("c1", cmdline.get<scalar_t>("c1")));
        }
        if (cmdline.has("c2"))
        {
            solver->config(nano::to_json("c2", cmdline.get<scalar_t>("c2")));
        }
        if (ls_init)
        {
            solver->lsearch(lsearch_init_t::all().get(*ls_init));
        }
        if (ls_strategy)
        {
            solver->lsearch(lsearch_strategy_t::all().get(*ls_strategy));
        }
        solver->epsilon(epsilon);
        solver->max_iterations(iterations);

        solvers.emplace_back(solver_id, std::move(solver));
    };

    for (const auto& id : solver_t::all().ids(sregex))
    {
        const auto size_init = ls_inits.size() + 1;
        const auto size_strategy = ls_strategies.size() + 1;

        for (size_t i_init = 0; i_init < size_init; ++ i_init)
        {
            for (size_t i_strategy = 0; i_strategy < size_strategy; ++ i_strategy)
            {
                add_solver(id,
                    (i_init == ls_inits.size()) ? nullptr : &ls_inits[i_init],
                    (i_strategy == ls_strategies.size()) ? nullptr : &ls_strategies[i_strategy]);
            }
        }
    }

    // benchmark
    solver_config_stats_t gstats;
    for (const auto& function : (is_convex ? get_convex_functions : get_functions)(min_dims, max_dims, fregex))
    {
        check_function(*function, solvers, trials, gstats);
    }

    show_table("Solver", gstats);

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
