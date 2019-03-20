#include <mutex>
#include "solver/gd.h"
#include "solver/cgd.h"
#include "solver/lbfgs.h"
#include "solver/quasi.h"
#include <nano/numeric.h>

using namespace nano;

solver_t::solver_t(const scalar_t c1, const scalar_t c2,
    const string_t& lsearch_init_id, const string_t& lsearch_strategy_id)
{
    lsearch_init(lsearch_init_id);
    lsearch_strategy(lsearch_strategy_id);

    m_lsearch_strategy->c1(c1);
    m_lsearch_strategy->c2(c2);
}

json_t solver_t::config() const
{
    const auto c1 = m_lsearch_strategy->c1();
    const auto c2 = m_lsearch_strategy->c2();

    json_t json;
    json["c1"] = strcat(c1, "(0,1)");
    json["c2"] = strcat(c2, "(c1,1)");
    json["init"] = m_lsearch_init->config_with_id(m_lsearch_init_id);
    json["strategy"] = m_lsearch_strategy->config_with_id(m_lsearch_strategy_id);

    return json;
}

void solver_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();

    auto c1 = m_lsearch_strategy->c1();
    auto c2 = m_lsearch_strategy->c2();
    nano::from_json_range(json, "c1", c1, eps, 1 - eps);
    nano::from_json_range(json, "c2", c2, c1, 1 - eps);

    if (json.count("init"))
    {
        lsearch_init(json["init"]);
    }
    if (json.count("strategy"))
    {
        lsearch_strategy(json["strategy"]);
    }

    m_lsearch_strategy->c1(c1);
    m_lsearch_strategy->c2(c2);
}

void solver_t::lsearch_init(const string_t& id)
{
    lsearch_init(id, lsearch_init_t::all().get(id));
}

void solver_t::lsearch_init(const string_t& id, rlsearch_init_t&& init)
{
    if (!init)
    {
        throw std::invalid_argument("invalid line-search initialization (" + id + ")");
    }

    m_lsearch_init_id = id;
    m_lsearch_init = std::move(init);
}

void solver_t::lsearch_init(const json_t& json)
{
    if (json.count("id") && m_lsearch_init_id != json["id"])
    {
        lsearch_init(json["id"]);
    }
    m_lsearch_init->config(json);
}

void solver_t::lsearch_strategy(const string_t& id)
{
    lsearch_strategy(id, lsearch_strategy_t::all().get(id));
}

void solver_t::lsearch_strategy(const string_t& id, rlsearch_strategy_t&& strategy)
{
    if (!strategy)
    {
        throw std::invalid_argument("invalid line-search strategy (" + id + ")");
    }

    const auto had_lsearch_strategy = static_cast<bool>(m_lsearch_strategy);
    const auto c1 = had_lsearch_strategy ? m_lsearch_strategy->c1() : scalar_t(0);
    const auto c2 = had_lsearch_strategy ? m_lsearch_strategy->c2() : scalar_t(0);

    m_lsearch_strategy_id = id;
    m_lsearch_strategy = std::move(strategy);

    // NB: keep the tolerances when changing the line-search strategy
    if (had_lsearch_strategy)
    {
        m_lsearch_strategy->c1(c1);
        m_lsearch_strategy->c2(c2);
    }
}

void solver_t::lsearch_strategy(const json_t& json)
{
    if (json.count("id") && m_lsearch_strategy_id != json["id"])
    {
        lsearch_strategy(json["id"]);
    }
    m_lsearch_strategy->config(json);
}

void solver_t::lsearch_logger(const lsearch_strategy_t::logger_t& logger)
{
    assert(m_lsearch_strategy);

    m_lsearch_strategy->logger(logger);
}

bool solver_t::lsearch(solver_state_t& state) const
{
    assert(m_lsearch_init);
    assert(m_lsearch_strategy);

    const auto t0 = m_lsearch_init->get(state);
    return m_lsearch_strategy->get(state, t0);
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

bool solver_t::log(solver_state_t& state) const
{
    const auto status = !m_logger ? true : m_logger(state);
    state.m_iterations ++;
    return status;
}

solver_factory_t& solver_t::all()
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
