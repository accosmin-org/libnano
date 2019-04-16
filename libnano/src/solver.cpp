#include <mutex>
#include "solver/gd.h"
#include "solver/cgd.h"
#include "solver/lbfgs.h"
#include "solver/quasi.h"
#include <nano/numeric.h>

using namespace nano;

solver_t::solver_t(const scalar_t c1, const scalar_t c2,
    const string_t& lsearch0_id, const string_t& lsearchk_id) :
    m_c1(c1),
    m_c2(c2)
{
    lsearch0(lsearch0_id);
    lsearchk(lsearchk_id);
}

json_t solver_t::config() const
{
    json_t json;
    json["c1"] = strcat(m_c1, "(0,1)");
    json["c2"] = strcat(m_c2, "(c1,1)");
    json["lsearch0"] = m_lsearch0->config_with_id(m_lsearch0_id);
    json["lsearchk"] = m_lsearchk->config_with_id(m_lsearchk_id);
    json["epsilon"] = strcat(m_epsilon, "(1e-12,1e-4)");
    json["max_iterations"] = strcat(m_max_iterations, "(1,1000000)");

    return json;
}

void solver_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();

    nano::from_json_range(json, "c1", m_c1, eps, 1 - eps);
    nano::from_json_range(json, "c2", m_c2, m_c1, 1 - eps);
    nano::from_json_range(json, "epsilon", m_epsilon, 1e-12 + eps, 1e-4 - eps);
    nano::from_json_range(json, "max_iterations", m_max_iterations, 1, 1000 * 1000);

    if (json.count("lsearch0"))
    {
        lsearch0(json["lsearch0"]);
    }
    if (json.count("lsearchk"))
    {
        lsearchk(json["lsearchk"]);
    }
}

void solver_t::lsearch0(const string_t& id)
{
    lsearch0(id, lsearch0_t::all().get(id));
}

void solver_t::lsearch0(const string_t& id, rlsearch0_t&& init)
{
    if (!init)
    {
        throw std::invalid_argument("invalid line-search initialization (" + id + ")");
    }

    m_lsearch0_id = id;
    m_lsearch0 = std::move(init);
}

void solver_t::lsearch0(const json_t& json)
{
    if (json.count("id") && m_lsearch0_id != json["id"])
    {
        lsearch0(json["id"]);
    }
    m_lsearch0->config(json);
}

void solver_t::lsearchk(const string_t& id)
{
    lsearchk(id, lsearchk_t::all().get(id));
}

void solver_t::lsearchk(const string_t& id, rlsearchk_t&& strategy)
{
    if (!strategy)
    {
        throw std::invalid_argument("invalid line-search strategy (" + id + ")");
    }

    m_lsearchk_id = id;
    m_lsearchk = std::move(strategy);
}

void solver_t::lsearchk(const json_t& json)
{
    if (json.count("id") && m_lsearchk_id != json["id"])
    {
        lsearchk(json["id"]);
    }
    m_lsearchk->config(json);
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

solver_state_t solver_t::minimize(const function_t& f, const vector_t& x0) const
{
    assert(f.size() == x0.size());

    // NB: create new line-search objects:
    //  - to have the solver thread-safe
    //  - to start with a fresh line-search history (needed for some strategies like CG_DESCENT)
    auto lsearch0 = lsearch0_t::all().get(m_lsearch0_id);
    lsearch0->epsilon(epsilon());
    lsearch0->logger(m_lsearch0_logger);
    lsearch0->config(m_lsearch0->config());

    auto lsearchk = lsearchk_t::all().get(m_lsearchk_id);
    lsearchk->c1(m_c1);
    lsearchk->c2(m_c2);
    lsearchk->logger(m_lsearchk_logger);
    lsearchk->config(m_lsearchk->config());

    auto function = solver_function_t{f};
    auto lsearch = lsearch_t{std::move(lsearch0), std::move(lsearchk)};

    return minimize(function, lsearch, x0);
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
