#include <mutex>
#include <nano/core/logger.h>
#include <nano/solver/asga.h>
#include <nano/solver/cgd.h>
#include <nano/solver/cocob.h>
#include <nano/solver/ellipsoid.h>
#include <nano/solver/gd.h>
#include <nano/solver/gs.h>
#include <nano/solver/lbfgs.h>
#include <nano/solver/osga.h>
#include <nano/solver/pdsgm.h>
#include <nano/solver/quasi.h>
#include <nano/solver/sgm.h>
#include <nano/solver/universal.h>

using namespace nano;

namespace
{
template <typename tsolver>
rsolver_t make_solver(const scalar_t epsilon, const tensor_size_t max_evals, const solver_t::logger_t& logger)
{
    auto solver = std::make_unique<tsolver>();
    solver->logger(logger);
    if (epsilon < 1e-7 && solver->type() == solver_type::line_search)
    {
        // NB: CG-DESCENT line-search gives higher precision than default More&Thuente line-search.
        solver->lsearchk("cgdescent");
    }
    solver->parameter("solver::epsilon")   = epsilon;
    solver->parameter("solver::max_evals") = max_evals;
    return solver;
}
} // namespace

solver_t::solver_t(string_t id)
    : clonable_t(std::move(id))
{
    lsearch0("quadratic");
    lsearchk("morethuente");

    register_parameter(parameter_t::make_scalar("solver::epsilon", 0, LT, 1e-8, LE, 1e-1));
    register_parameter(parameter_t::make_integer("solver::max_evals", 10, LE, 1000, LE, 1e+9));
    register_parameter(parameter_t::make_scalar_pair("solver::tolerance", 0, LT, 1e-4, LT, 0.1, LT, 1));
}

solver_t::solver_t(const solver_t& other)
    : configurable_t(other)
    , clonable_t(other)
    , m_logger(other.m_logger)
    , m_lsearch0(other.lsearch0().clone())
    , m_lsearchk(other.lsearchk().clone())
    , m_type(other.type())
{
}

void solver_t::lsearch0(const string_t& id)
{
    auto lsearch0 = lsearch0_t::all().get(id);
    critical(!lsearch0, "invalid lsearch0 id <", id, ">!");

    m_lsearch0 = std::move(lsearch0);
}

void solver_t::lsearch0(const lsearch0_t& lsearch0)
{
    m_lsearch0 = lsearch0.clone();
}

void solver_t::lsearchk(const string_t& id)
{
    auto lsearchk = lsearchk_t::all().get(id);
    critical(!lsearchk, "invalid lsearchk id <", id, ">!");

    m_lsearchk = std::move(lsearchk);
}

void solver_t::lsearchk(const lsearchk_t& lsearchk)
{
    m_lsearchk = lsearchk.clone();
}

void solver_t::lsearch0_logger(const lsearch0_t::logger_t& logger)
{
    m_lsearch0->logger(logger);
}

void solver_t::lsearchk_logger(const lsearchk_t::logger_t& logger)
{
    m_lsearchk->logger(logger);
}

void solver_t::logger(const logger_t& logger)
{
    m_logger = logger;
}

void solver_t::type(const solver_type type)
{
    m_type = type;
}

solver_type solver_t::type() const
{
    return m_type;
}

lsearch_t solver_t::make_lsearch() const
{
    // NB: create new line-search objects:
    //  - to have the solver thread-safe
    //  - to start with a fresh line-search history
    //  - to pass the solver's parameters
    auto lsearch0 = m_lsearch0->clone();
    auto lsearchk = m_lsearchk->clone();

    lsearch0->parameter("lsearch0::epsilon")   = parameter("solver::epsilon").value<scalar_t>();
    lsearchk->parameter("lsearchk::tolerance") = parameter("solver::tolerance").value_pair<scalar_t>();
    return lsearch_t{std::move(lsearch0), std::move(lsearchk)};
}

solver_state_t solver_t::minimize(const function_t& function, const vector_t& x0) const
{
    critical(function.size() != x0.size(), "solver: incompatible initial point (", x0.size(),
             " dimensions), expecting ", function.size(), " dimensions!");

    function.clear_statistics();

    return do_minimize(function, x0);
}

bool solver_t::done(solver_state_t& state, const bool iter_ok, const bool converged) const
{
    // stopping was requested (in an outer loop)
    if (state.status() == solver_status::stopped)
    {
        return true;
    }
    else
    {
        if (const auto step_ok = iter_ok && state.valid(); converged || !step_ok)
        {
            // either converged or failed
            state.status(converged ? solver_status::converged : solver_status::failed);
            log(state);
            return true;
        }
        else if (!log(state))
        {
            // stopping was requested
            state.status(solver_status::stopped);
            return true;
        }

        // OK, go on with the optimization
        return false;
    }
}

bool solver_t::log(const solver_state_t& state) const
{
    return !m_logger ? true : m_logger(state);
}

factory_t<solver_t>& solver_t::all()
{
    static auto manager = factory_t<solver_t>{};
    const auto  op      = []()
    {
        manager.add<solver_gd_t>("gradient descent");
        manager.add<solver_gs_t>("gradient sampling");
        manager.add<solver_sgm_t>("sub-gradient method");
        manager.add<solver_cgd_pr_t>("conjugate gradient descent (default)");
        manager.add<solver_cgd_n_t>("conjugate gradient descent (N+)");
        manager.add<solver_cgd_hs_t>("conjugate gradient descent (HS+)");
        manager.add<solver_cgd_fr_t>("conjugate gradient descent (FR)");
        manager.add<solver_cgd_pr_t>("conjugate gradient descent (PR+)");
        manager.add<solver_cgd_cd_t>("conjugate gradient descent (CD)");
        manager.add<solver_cgd_ls_t>("conjugate gradient descent (LS+)");
        manager.add<solver_cgd_dy_t>("conjugate gradient descent (DY)");
        manager.add<solver_cgd_dycd_t>("conjugate gradient descent (DYCD)");
        manager.add<solver_cgd_dyhs_t>("conjugate gradient descent (DYHS)");
        manager.add<solver_cgd_frpr_t>("conjugate gradient descent (FRPR)");
        manager.add<solver_osga_t>("optimal sub-gradient algorithm (OSGA)");
        manager.add<solver_lbfgs_t>("limited-memory BFGS");
        manager.add<solver_quasi_dfp_t>("quasi-newton method (DFP)");
        manager.add<solver_quasi_sr1_t>("quasi-newton method (SR1)");
        manager.add<solver_quasi_bfgs_t>("quasi-newton method (BFGS)");
        manager.add<solver_quasi_hoshino_t>("quasi-newton method (Hoshino formula)");
        manager.add<solver_quasi_fletcher_t>("quasi-newton method (Fletcher's switch)");
        manager.add<solver_ellipsoid_t>("ellipsoid method");
        manager.add<solver_asga2_t>("accelerated sub-gradient algorithm (ASGA-2)");
        manager.add<solver_asga4_t>("accelerated sub-gradient algorithm (ASGA-4)");
        manager.add<solver_cocob_t>("continuous coin betting (COCOB)");
        manager.add<solver_sda_t>("simple dual averages (variant of primal-dual subgradient methods)");
        manager.add<solver_wda_t>("weighted dual averages (variant of primal-dual subgradient methods)");
        manager.add<solver_pgm_t>("universal primal gradient method (PGM)");
        manager.add<solver_dgm_t>("universal dual gradient method (DGM)");
        manager.add<solver_fgm_t>("universal fast gradient method (FGM)");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}

rsolver_t solver_t::make_solver(const function_t& function, const scalar_t epsilon, const tensor_size_t max_evals) const
{
    return function.smooth() ? ::make_solver<solver_lbfgs_t>(epsilon, max_evals, m_logger)
                             : ::make_solver<solver_osga_t>(epsilon, max_evals, m_logger);
}

void solver_t::more_precise(const scalar_t epsilon_factor)
{
    assert(0.0 < epsilon_factor && epsilon_factor < 1.0);

    parameter("solver::epsilon") = parameter("solver::epsilon").value<scalar_t>() * epsilon_factor;
}
