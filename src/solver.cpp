#include <mutex>
#include <nano/critical.h>
#include <solver/asga.h>
#include <solver/cgd.h>
#include <solver/cocob.h>
#include <solver/ellipsoid.h>
#include <solver/fpba.h>
#include <solver/gd.h>
#include <solver/gsample.h>
#include <solver/interior.h>
#include <solver/lbfgs.h>
#include <solver/osga.h>
#include <solver/pdsgm.h>
#include <solver/quasi.h>
#include <solver/rqb.h>
#include <solver/sgm.h>
#include <solver/universal.h>

using namespace nano;

namespace
{
template <class tsolver>
rsolver_t make_solver(const scalar_t epsilon, const tensor_size_t max_evals)
{
    auto solver = std::make_unique<tsolver>();
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
    : typed_t(std::move(id))
{
    lsearch0("quadratic");
    lsearchk("cgdescent");

    register_parameter(parameter_t::make_scalar("solver::epsilon", 0, LT, 1e-8, LE, 1e-1));
    register_parameter(parameter_t::make_integer("solver::max_evals", 10, LE, 1000, LE, 1e+9));
    register_parameter(parameter_t::make_scalar_pair("solver::tolerance", 0, LT, 1e-4, LT, 0.1, LT, 1));
}

solver_t::solver_t(const solver_t& other)
    : typed_t(other)
    , configurable_t(other)
    , clonable_t(other)
    , m_lsearch0(other.lsearch0().clone())
    , m_lsearchk(other.lsearchk().clone())
    , m_type(other.type())
{
}

void solver_t::lsearch0(const string_t& id)
{
    auto lsearch0 = lsearch0_t::all().get(id);
    critical(lsearch0, "invalid lsearch0 id <", id, ">!");

    m_lsearch0 = std::move(lsearch0);
}

void solver_t::lsearch0(const lsearch0_t& lsearch0)
{
    m_lsearch0 = lsearch0.clone();
}

void solver_t::lsearchk(const string_t& id)
{
    auto lsearchk = lsearchk_t::all().get(id);
    critical(lsearchk, "invalid lsearchk id <", id, ">!");

    m_lsearchk = std::move(lsearchk);
}

void solver_t::lsearchk(const lsearchk_t& lsearchk)
{
    m_lsearchk = lsearchk.clone();
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

solver_state_t solver_t::minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    critical(function.size() == x0.size(), "solver: incompatible initial point (", x0.size(),
             " dimensions), expecting ", function.size(), " dimensions!");

    function.clear_statistics();

    // TODO: check compatibility with the function, warn otherwise!

    return do_minimize(function, x0, logger);
}

bool solver_t::done(solver_state_t& state, const bool iter_ok, const bool converged, const logger_t& logger) const
{
    state.update_calls();

    if (state.status() == solver_status::unfeasible || state.status() == solver_status::unbounded)
    {
        // unfeasible constrained problem
        logger.info("[solver-", type_id(), "]: ", state, ".\n");
        return true;
    }
    else if (const auto step_ok = iter_ok && state.valid(); converged || !step_ok)
    {
        // either converged or failed
        state.status(converged ? solver_status::converged : solver_status::failed);
        logger.info("[solver-", type_id(), "]: ", state, ".\n");
        return true;
    }
    else
    {
        // OK, go on with the optimization
        logger.info("[solver-", type_id(), "]: ", state, ".\n");
        return false;
    }
}

factory_t<solver_t>& solver_t::all()
{
    static auto manager = factory_t<solver_t>{};
    const auto  op      = []()
    {
        manager.add<solver_gd_t>("gradient descent");
        manager.add<solver_gs_t>("gradient sampling (P-nNGS)");
        manager.add<solver_ags_t>("adaptive gradient sampling (P-nNGS + AGS)");
        manager.add<solver_gs_lbfgs_t>("gradient sampling with LBFGS-like updates (P-nNGS + LBFGS)");
        manager.add<solver_ags_lbfgs_t>("adaptive gradient sampling with LBFGS-like updates (P-nNGS + AGS + LBFGS)");
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
        manager.add<solver_rqb_t>("reversal quasi-newton bundle algorithm (RQB)");
        manager.add<solver_fpba1_t>("fast proximal bundle algorithm (FPBA1)");
        manager.add<solver_fpba2_t>("fast proximal bundle algorithm (FPBA2)");
        manager.add<solver_ipm_t>("primal-dual interior point method for linear and quadratic programs (IPM)");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}

rsolver_t solver_t::make_solver(const function_t& function, const scalar_t epsilon, const tensor_size_t max_evals)
{
    // FIXME: should use RQB or some other proximal bundle method
    return function.smooth() ? ::make_solver<solver_lbfgs_t>(epsilon, max_evals)
                             : ::make_solver<solver_osga_t>(epsilon, max_evals);
}

void solver_t::more_precise(const scalar_t epsilon_factor)
{
    assert(0.0 < epsilon_factor && epsilon_factor < 1.0);

    parameter("solver::epsilon") = parameter("solver::epsilon").value<scalar_t>() * epsilon_factor;
}
