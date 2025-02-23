#include <filesystem>
#include <iomanip>
#include <nano/core/chrono.h>
#include <nano/core/cmdline.h>
#include <nano/core/numeric.h>
#include <nano/core/parallel.h>
#include <nano/core/table.h>
#include <nano/critical.h>
#include <nano/main.h>
#include <nano/solver.h>

using namespace nano;

namespace
{
struct result_t
{
    result_t() = default;

    result_t(const solver_state_t& state, const int64_t milliseconds)
        : m_value(state.fx())
        , m_gtest(state.gradient_test())
        , m_ktest(state.kkt_optimality_test())
        , m_etest(state.kkt_optimality_test2())
        , m_itest(state.kkt_optimality_test1())
        , m_status(state.status())
        , m_fcalls(state.fcalls())
        , m_gcalls(state.gcalls())
        , m_milliseconds(milliseconds)
    {
    }

    scalar_t      m_value{0.0};                    ///< objective value
    scalar_t      m_gtest{0.0};                    ///< gradient test
    scalar_t      m_ktest{0.0};                    ///< kkt optimality test
    scalar_t      m_etest{0.0};                    ///< equality contraints violation
    scalar_t      m_itest{0.0};                    ///< inequality constraints violation
    solver_status m_status{solver_status::failed}; ///<
    tensor_size_t m_fcalls{0};                     ///<
    tensor_size_t m_gcalls{0};                     ///<
    int64_t       m_milliseconds{0};               ///<
};

struct solver_stats_t
{
    explicit solver_stats_t(const size_t trials)
        : m_stats(12, static_cast<tensor_size_t>(trials))
    {
    }

    // clang-format off

    ///< objective values
    auto values() { return m_stats.tensor(0); }
    auto values() const { return m_stats.tensor(0); }

    ///< gradient tests
    auto gtests() { return m_stats.tensor(1); }
    auto gtests() const { return m_stats.tensor(1); }

    ///< kkt optimality tests
    auto ktests() { return m_stats.tensor(2); }
    auto ktests() const { return m_stats.tensor(2); }

    ///< equality constraints violations
    auto etests() { return m_stats.tensor(3); }
    auto etests() const { return m_stats.tensor(3); }

    ///< inequality constraints violations
    auto itests() { return m_stats.tensor(4); }
    auto itests() const { return m_stats.tensor(4); }

    ///< #internal errors (e.g. line-search failed)
    auto errors() { return m_stats.tensor(5); }
    auto errors() const { return m_stats.tensor(5); }

    ///< #maximum iterations reached (without convergence)
    auto maxits() { return m_stats.tensor(6); }
    auto maxits() const { return m_stats.tensor(6); }

    ///< #function value calls
    auto fcalls() { return m_stats.tensor(7); }
    auto fcalls() const { return m_stats.tensor(7); }

    ///< #gradient calls
    auto gcalls() { return m_stats.tensor(8); }
    auto gcalls() const { return m_stats.tensor(8); }

    ///< number of milliseconds
    auto millis() { return m_stats.tensor(9); }
    auto millis() const { return m_stats.tensor(9); }

    ///< rank as ordered by the function value
    auto ranks() { return m_stats.tensor(10); }
    auto ranks() const { return m_stats.tensor(10); }

    ///< relative precision to the best solver
    auto precisions() { return m_stats.tensor(11); }
    auto precisions() const { return m_stats.tensor(11); }

    // clang-format on

    void update(const size_t trial, const result_t& result, const scalar_t precision, const ptrdiff_t rank)
    {
        const auto itrial    = static_cast<tensor_size_t>(trial);
        values()(itrial)     = result.m_value;
        gtests()(itrial)     = result.m_gtest;
        ktests()(itrial)     = result.m_ktest;
        etests()(itrial)     = result.m_etest;
        itests()(itrial)     = result.m_itest;
        errors()(itrial)     = (result.m_status == solver_status::failed) ? 1.0 : 0.0;
        maxits()(itrial)     = (result.m_status == solver_status::max_iters) ? 1.0 : 0.0;
        fcalls()(itrial)     = static_cast<scalar_t>(result.m_fcalls);
        gcalls()(itrial)     = static_cast<scalar_t>(result.m_gcalls);
        millis()(itrial)     = static_cast<scalar_t>(result.m_milliseconds);
        ranks()(itrial)      = static_cast<scalar_t>(rank);
        precisions()(itrial) = precision;
    }

    void update(const size_t trial, const solver_stats_t& stats)
    {
        const auto itrial    = static_cast<tensor_size_t>(trial);
        values()(itrial)     = std::numeric_limits<scalar_t>::quiet_NaN();
        gtests()(itrial)     = stats.gtests().mean();
        ktests()(itrial)     = stats.ktests().mean();
        etests()(itrial)     = stats.etests().mean();
        itests()(itrial)     = stats.itests().mean();
        errors()(itrial)     = stats.errors().sum();
        maxits()(itrial)     = stats.maxits().sum();
        fcalls()(itrial)     = stats.fcalls().mean();
        gcalls()(itrial)     = stats.gcalls().mean();
        millis()(itrial)     = stats.millis().mean();
        ranks()(itrial)      = stats.ranks().mean();
        precisions()(itrial) = stats.precisions().mean();
    }

    // attributes
    tensor2d_t m_stats; ///< statistics (statistic index, #trials)
};

auto relative_precision(const scalar_t value, const scalar_t best_value, const scalar_t epsilon)
{
    assert(value >= best_value);
    return std::log10(std::max(value - best_value, epsilon));
}

auto relative_precision(const result_t& result, const result_t& best_result, const scalar_t epsilon,
                        const function_type fun_type)
{
    switch (fun_type)
    {
    case function_type::convex:
    case function_type::smooth:
    case function_type::convex_smooth:
    case function_type::convex_nonsmooth:
        return relative_precision(result.m_value, best_result.m_value, epsilon);

    default:
        return relative_precision(result.m_ktest, best_result.m_ktest, epsilon);
    }
}

auto make_solver_name(const rsolver_t& solver)
{
    return solver->has_lsearch()
             ? scat(solver->type_id(), " [", solver->lsearch0().type_id(), ",", solver->lsearchk().type_id(), "]")
             : solver->type_id();
}

using points_t = std::vector<vector_t>;
using results_t = std::vector<result_t>;
using solvers_t = std::vector<rsolver_t>;

auto& print_scalar(row_t& row, const scalar_t value)
{
    return std::isfinite(value) ? (row << value) : (row << "N/A");
}

auto& print_integer(row_t& row, const scalar_t value)
{
    return std::isfinite(value) ? (row << static_cast<size_t>(value)) : (row << "N/A");
}

void print_table(string_t table_name, const solvers_t& solvers, const std::vector<solver_stats_t>& stats,
                 const function_type fun_type)
{
    // gather statistics per solver
    const auto max_evals        = solvers[0U]->parameter("solver::max_evals").value<int>();
    const auto max_digits_calls = static_cast<size_t>(std::log10(max_evals)) + 1U;

    if (table_name.size() > 32U)
    {
        table_name = table_name.substr(0U, 24U) + "..." + table_name.substr(table_name.size() - 5U, 5U);
    }

    // display per-function statistics
    table_t table;
    auto&   header = table.header();
    header << align(table_name, 32) << align("precision", 9) << align("rank", 4) << align("value", 12);
    switch (fun_type)
    {
    case function_type::smooth:
    case function_type::convex_smooth:
        header << align("grad test", 12);
        break;

    case function_type::linear_program:
    case function_type::quadratic_program:
        header << align("kkt test", 12);
        break;

    default:
        break;
    }
    header << "errors"
           << "maxits" << align("fcalls", max_digits_calls) << align("gcalls", max_digits_calls) << align("[ms]", 5);
    table.delim();

    for (size_t isolver = 0U; isolver < solvers.size(); ++isolver)
    {
        const auto& stat        = stats[isolver];
        const auto  solver_name = make_solver_name(solvers[isolver]);

        auto& row = table.append();
        row << solver_name << scat(std::fixed, std::setprecision(4), stat.precisions().mean())
            << scat(std::fixed, std::setprecision(2), stat.ranks().mean());
        print_scalar(row, stat.values().mean());

        switch (fun_type)
        {
        case function_type::smooth:
        case function_type::convex_smooth:
            print_scalar(row, stat.gtests().mean());
            break;

        case function_type::linear_program:
        case function_type::quadratic_program:
            print_scalar(row, stat.ktests().mean());
            break;

        default:
            break;
        }

        print_integer(row, stat.errors().sum());
        print_integer(row, stat.maxits().sum());
        print_integer(row, stat.fcalls().mean());
        print_integer(row, stat.gcalls().mean());
        print_integer(row, stat.millis().mean());
    }

    // NB: sort solvers by the average precision!
    assert(table.rows() == solvers.size() + 2U);
    table.sort(nano::make_less_from_string<scalar_t>(), {1});
    std::cout << table;
}

auto minimize_all(parallel::pool_t& pool, const function_t& function, const solvers_t& solvers, const points_t& x0s,
                  const string_t& log_dir)
{
    const auto minimize_one = [&](const size_t i)
    {
        const auto itrial  = i / solvers.size();
        const auto isolver = i % solvers.size();

        const auto& x0     = x0s[itrial];
        const auto  solver = solvers[isolver]->clone();

        const auto logger = [&]()
        {
            if (!log_dir.empty())
            {
                const auto path = std::filesystem::path(log_dir) / function.name() / scat("trial", itrial + 1) /
                                  scat(solver->type_id(), ".log");

                return make_file_logger(path.string());
            }
            else
            {
                return make_null_logger();
            }
        }();

        const auto timer = nano::timer_t{};
        const auto state = solver->minimize(*function.clone(), x0, logger);
        const auto milli = timer.milliseconds().count();

        return result_t{state, milli};
    };

    auto results = results_t{x0s.size() * solvers.size()};
    pool.map(results.size(), [&](const size_t i, const size_t) { results[i] = minimize_one(i); });
    return results;
}

auto benchmark(parallel::pool_t& pool, const function_t& function, const solvers_t& solvers, const size_t trials,
               const string_t& log_dir, const function_type fun_type)
{
    // generate a fixed set of random initial points
    points_t x0s(trials);
    std::generate(x0s.begin(), x0s.end(), [&]() { return make_random_vector<scalar_t>(function.size()); });

    // and minimize in parallel all (solver, random initial point) combinations
    const auto results = minimize_all(pool, function, solvers, x0s, log_dir);

    // gather statistics per solver
    auto stats = std::vector<solver_stats_t>{solvers.size(), solver_stats_t{trials}};
    auto ranks = std::vector<std::pair<scalar_t, size_t>>{solvers.size()};

    for (size_t trial = 0U; trial < trials; ++trial)
    {
        const auto* const begin = &results[trial * solvers.size()];
        const auto* const end   = &results[trial * solvers.size() + solvers.size()];

        const auto op = [&](const auto& lhs, const auto& rhs)
        {
            switch (fun_type)
            {
            case function_type::smooth:
            case function_type::convex_smooth:
                return lhs.m_gtest < rhs.m_gtest;

            case function_type::convex:
            case function_type::convex_nonsmooth:
                return lhs.m_value < rhs.m_value;

            default:
                return lhs.m_ktest < rhs.m_ktest;
            }
        };

        const auto& best_result = *std::min_element(begin, end, op);

        for (size_t isolver = 0U; isolver < solvers.size(); ++isolver)
        {
            const auto& result = results[trial * solvers.size() + isolver];
            ranks[isolver]     = std::make_pair(result.m_value, isolver);
        }
        std::sort(ranks.begin(), ranks.end());

        for (size_t isolver = 0U; isolver < solvers.size(); ++isolver)
        {
            const auto& result = results[trial * solvers.size() + isolver];
            assert(std::isfinite(result.m_value));
            assert(std::isfinite(result.m_gtest));

            const auto epsilon   = solvers[isolver]->parameter("solver::epsilon").value<scalar_t>();
            const auto precision = relative_precision(result, best_result, epsilon, fun_type);

            const auto find = [&](const auto& v) { return v.second == isolver; };
            const auto rank = (std::find_if(ranks.begin(), ranks.end(), find) - ranks.begin()) + 1;

            stats[isolver].update(trial, result, precision, rank);
        }
    }

    // display per-function statistics
    print_table(function.name(), solvers, stats, fun_type);

    return stats;
}

int unsafe_main(int argc, const char* argv[])
{
    using namespace nano;

    // parse the command line
    cmdline_t cmdline("benchmark solvers on unconstrained nonlinear problems");
    cmdline.add("--solver", "regex to select solvers", ".+");
    cmdline.add("--function", "regex to select test functions", ".+");
    cmdline.add("--lsearch0", "regex to select line-search initialization methods", "quadratic");
    cmdline.add("--lsearchk", "regex to select line-search strategies", "cgdescent");
    cmdline.add("--min-dims", "minimum number of dimensions for each test function (if feasible)", "4");
    cmdline.add("--max-dims", "maximum number of dimensions for each test function (if feasible)", "16");
    cmdline.add("--trials", "number of random trials for each test function", "100");
    cmdline.add(
        "--function-type",
        "function type, one of [convex, smooth, convex-smooth, convex-nonsmooth, linear-program, quadratic-program]",
        "convex-smooth");
    cmdline.add("--log-dir", "directory to log the optimization trajectories");

    const auto options = cmdline.process(argc, argv);
    if (cmdline.handle(options))
    {
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto min_dims = options.get<tensor_size_t>("--min-dims");
    const auto max_dims = options.get<tensor_size_t>("--max-dims");
    const auto trials   = options.get<size_t>("--trials");
    const auto fun_type = nano::from_string<function_type>(options.get<string_t>("--function-type"));
    const auto log_dir = options.has("--log-dir") ? options.get("--log-dir") : string_t{};
    const auto fregex  = std::regex(options.get<string_t>("--function"));
    const auto sregex  = std::regex(options.get<string_t>("--solver"));
    const auto l0regex = std::regex(options.get<string_t>("--lsearch0"));
    const auto lkregex = std::regex(options.get<string_t>("--lsearchk"));
    const auto lsearch0_ids = options.has("--lsearch0") ? lsearch0_t::all().ids(l0regex) : strings_t{""};
    const auto lsearchk_ids = options.has("--lsearchk") ? lsearchk_t::all().ids(lkregex) : strings_t{""};

    // clang-format off
    critical(
        fun_type == function_type::convex ||
        fun_type == function_type::smooth ||
        fun_type == function_type::convex_smooth ||
        fun_type == function_type::convex_nonsmooth ||
        fun_type == function_type::linear_program ||
        fun_type == function_type::quadratic_program,
        "unsupported function type!");
    // clang-format on

    const auto solver_ids = solver_t::all().ids(sregex);
    critical(!solver_ids.empty(), "at least a solver needs to be selected!");

    const auto functions = function_t::make({min_dims, max_dims, fun_type}, fregex);
    critical(!functions.empty(), "at least a function needs to be selected!");

    auto rconfig = cmdconfig_t{options};

    // construct the list of solver configurations to evaluate
    solvers_t solvers;
    for (const auto& solver_id : solver_ids)
    {
        auto solver = solver_t::all().get(solver_id);
        if (solver->has_lsearch())
        {
            for (const auto& lsearch0_id : lsearch0_ids)
            {
                for (const auto& lsearchk_id : lsearchk_ids)
                {
                    solver        = solver_t::all().get(solver_id);
                    auto lsearch0 = lsearch0_t::all().get(lsearch0_id);
                    auto lsearchk = lsearchk_t::all().get(lsearchk_id);

                    rconfig.setup(*solver);
                    rconfig.setup(*lsearch0);
                    rconfig.setup(*lsearchk);

                    solver->lsearch0(*lsearch0);
                    solver->lsearchk(*lsearchk);

                    solvers.emplace_back(std::move(solver));
                }
            }
        }
        else
        {
            rconfig.setup(*solver);
            solvers.emplace_back(std::move(solver));
        }
    }

    // benchmark solvers and display statistics independently per function
    auto thread_pool  = parallel::pool_t{};
    auto solver_stats = std::vector<solver_stats_t>{solvers.size(), solver_stats_t{functions.size()}};

    for (size_t ifunction = 0U; ifunction < functions.size(); ++ifunction)
    {
        const auto& function = functions[ifunction];
        const auto  funstats = benchmark(thread_pool, *function, solvers, trials, log_dir, fun_type);
        for (size_t isolver = 0U; isolver < solvers.size(); ++isolver)
        {
            const auto& stats = funstats[isolver];
            solver_stats[isolver].update(ifunction, stats);
        }
    }

    // display global statistics
    print_table("solver", solvers, solver_stats, fun_type);

    // OK
    return EXIT_SUCCESS;
}
} // namespace

int main(int argc, const char* argv[])
{
    return nano::safe_main(unsafe_main, argc, argv);
}
