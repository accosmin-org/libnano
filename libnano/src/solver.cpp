#include <mutex>
#include "solver/gd.h"
#include "solver/cgd.h"
#include "solver/lbfgs.h"
#include "solver/quasi.h"
#include <nano/numeric.h>

using namespace nano;

solver_factory_t& nano::get_solvers()
{
    static solver_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<solver_gd_t>("gd", "gradient descent");
        manager.add<solver_cgd_prp_t>("cgd", "conjugate gradient descent (default)");
        manager.add<solver_cgd_n_t>("cgd-n", "conjugate gradient descent (N)");
        manager.add<solver_cgd_hs_t>("cgd-hs", "conjugate gradient descent (HS)");
        manager.add<solver_cgd_fr_t>("cgd-fr", "conjugate gradient descent (FR)");
        manager.add<solver_cgd_prp_t>("cgd-prp", "conjugate gradient descent (PRP+)");
        manager.add<solver_cgd_cd_t>("cgd-cd", "conjugate gradient descent (CD)");
        manager.add<solver_cgd_ls_t>("cgd-ls", "conjugate gradient descent (LS)");
        manager.add<solver_cgd_dy_t>("cgd-dy", "conjugate gradient descent (DY)");
        manager.add<solver_cgd_dycd_t>("cgd-dycd", "conjugate gradient descent (DYCD)");
        manager.add<solver_cgd_dyhs_t>("cgd-dyhs", "conjugate gradient descent (DYHS)");
        manager.add<solver_lbfgs_t>("lbfgs", "limited-memory BFGS");
        manager.add<solver_quasi_dfp_t>("dfp", "quasi-newton method (DFP)");
        manager.add<solver_quasi_sr1_t>("sr1", "quasi-newton method (SR1)");
        manager.add<solver_quasi_bfgs_t>("bfgs", "quasi-newton method (BFGS)");
        manager.add<solver_quasi_broyden_t>("broyden", "quasi-newton method (Broyden)");
    });

    return manager;
}

solver_t::solver_t(const scalar_t c1, const scalar_t c2) :
    m_lsearch_init(get_lsearch_inits().get("quadratic")),
    m_lsearch_algo(get_lsearch_strategies().get("morethuente"))
{
    if (!m_lsearch_init) throw std::runtime_error("invalid line-search initialization");
    if (!m_lsearch_algo) throw std::runtime_error("invalid line-search strategy");

    m_lsearch_algo->c1(c1);
    m_lsearch_algo->c2(c2);
}

void solver_t::to_json(json_t& json) const
{
    const auto c1 = m_lsearch_algo->c1();
    const auto c2 = m_lsearch_algo->c2();

    nano::to_json(json,
        "c1", strcat(c1, "(0,1)"),
        "c2", strcat(c2, "(c1,1)"));
}

void solver_t::from_json(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();

    auto c1 = m_lsearch_algo->c1();
    auto c2 = m_lsearch_algo->c2();
    nano::from_json_range(json, "c1", c1, eps, 1 - eps);
    nano::from_json_range(json, "c2", c2, c1, 1 - eps);

    m_lsearch_algo->c1(c1);
    m_lsearch_algo->c2(c2);
}

void solver_t::lsearch(rlsearch_init_t&& init)
{
    if (!init) throw std::runtime_error("invalid line-search initialization");

    m_lsearch_init = std::move(init);
}

void solver_t::lsearch(rlsearch_strategy_t&& algo)
{
    if (!algo) throw std::runtime_error("invalid line-search strategy");

    m_lsearch_algo = std::move(algo);
}

bool solver_t::lsearch(solver_state_t& state) const
{
    assert(m_lsearch_init);
    assert(m_lsearch_algo);

    // check descent direction
    if (!state.has_descent())
    {
        return false;
    }

    // initial step length
    // fixme: is it safe to allow t0 to be greater than 1?!
    const auto t0 = nano::clamp(m_lsearch_init->get(state), lsearch_strategy_t::stpmin(), scalar_t(1));

    // line-search step length
    auto state0 = state;
    return m_lsearch_algo->get(state0, t0, state) && state && (state < state0);
}

bool solver_t::done(const solver_function_t& function, solver_state_t& state, const bool iter_ok) const
{
    state.m_fcalls = function.fcalls();
    state.m_gcalls = function.gcalls();

    const auto step_ok = iter_ok && state;
    const auto converged = state.converged(epsilon());

    if (converged || !step_ok)
    {
        // either converged or failed
        state.m_status = converged ?
            solver_state_t::status::converged :
            solver_state_t::status::failed;
        log(state);
        return true;
    }
    else if (!log(state))
    {
        // stopping was requested
        state.m_status = solver_state_t::status::stopped;
        return true;
    }

    // OK, go on with the optimization
    return false;
}
