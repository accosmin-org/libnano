#include <mutex>
#include <nano/solver/gd.h>
#include <nano/solver/cgd.h>
#include <nano/solver/lbfgs.h>
#include <nano/solver/quasi.h>

using namespace nano;

lsearch_solver_t::lsearch_solver_t(const scalar_t c1, const scalar_t c2,
    const string_t& lsearch0_id, const string_t& lsearchk_id)
{
    lsearch0(lsearch0_id);
    lsearchk(lsearchk_id);
    tolerance(c1, c2);
}

void lsearch_solver_t::lsearch0(const string_t& id)
{
    lsearch0(id, lsearch0_t::all().get(id));
}

void lsearch_solver_t::lsearch0(const string_t& id, rlsearch0_t&& init)
{
    if (!init)
    {
        throw std::invalid_argument("invalid line-search initialization (" + id + ")");
    }

    m_lsearch0_id = id;
    m_lsearch0 = std::move(init);
}

void lsearch_solver_t::lsearchk(const string_t& id)
{
    lsearchk(id, lsearchk_t::all().get(id));
}

void lsearch_solver_t::lsearchk(const string_t& id, rlsearchk_t&& strategy)
{
    if (!strategy)
    {
        throw std::invalid_argument("invalid line-search strategy (" + id + ")");
    }

    m_lsearchk_id = id;
    m_lsearchk = std::move(strategy);
}

void lsearch_solver_t::lsearch0_logger(const lsearch0_t::logger_t& logger)
{
    m_lsearch0->logger(logger);
}

void lsearch_solver_t::lsearchk_logger(const lsearchk_t::logger_t& logger)
{
    m_lsearchk->logger(logger);
}

void lsearch_solver_t::tolerance(const scalar_t c1, const scalar_t c2)
{
    m_lsearchk->tolerance(c1, c2);
}

solver_state_t lsearch_solver_t::minimize(const function_t& f, const vector_t& x0) const
{
    assert(f.size() == x0.size());

    // NB: create new line-search objects:
    //  - to have the solver thread-safe
    //  - to start with a fresh line-search history (needed for some strategies like CG_DESCENT)
    auto lsearch0 = m_lsearch0->clone();
    auto lsearchk = m_lsearchk->clone();

    lsearch0->epsilon(epsilon());

    auto function = solver_function_t{f};
    auto lsearch = lsearch_t{std::move(lsearch0), std::move(lsearchk)};

    return iterate(function, lsearch, x0);
}

lsearch_solver_factory_t& lsearch_solver_t::all()
{
    static lsearch_solver_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<solver_gd_t>("gd", "gradient descent");
        manager.add<solver_cgd_pr_t>("cgd", "conjugate gradient descent (default)");
        manager.add<solver_cgd_n_t>("cgd-n", "conjugate gradient descent (N+)");
        manager.add<solver_cgd_hs_t>("cgd-hs", "conjugate gradient descent (HS+)");
        manager.add<solver_cgd_fr_t>("cgd-fr", "conjugate gradient descent (FR)");
        manager.add<solver_cgd_pr_t>("cgd-pr", "conjugate gradient descent (PR+)");
        manager.add<solver_cgd_cd_t>("cgd-cd", "conjugate gradient descent (CD)");
        manager.add<solver_cgd_ls_t>("cgd-ls", "conjugate gradient descent (LS+)");
        manager.add<solver_cgd_dy_t>("cgd-dy", "conjugate gradient descent (DY)");
        manager.add<solver_cgd_dycd_t>("cgd-dycd", "conjugate gradient descent (DYCD)");
        manager.add<solver_cgd_dyhs_t>("cgd-dyhs", "conjugate gradient descent (DYHS)");
        manager.add<solver_cgd_frpr_t>("cgd-prfr", "conjugate gradient descent (FRPR)");
        manager.add<solver_lbfgs_t>("lbfgs", "limited-memory BFGS");
        manager.add<solver_quasi_dfp_t>("dfp", "quasi-newton method (DFP)");
        manager.add<solver_quasi_sr1_t>("sr1", "quasi-newton method (SR1)");
        manager.add<solver_quasi_bfgs_t>("bfgs", "quasi-newton method (BFGS)");
        manager.add<solver_quasi_hoshino_t>("hoshino", "quasi-newton method (Hoshino formula)");
        manager.add<solver_quasi_fletcher_t>("fletcher", "quasi-newton method (Fletcher's switch)");
    });

    return manager;
}
