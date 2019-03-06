#include <mutex>
#include <nano/numeric.h>

#include "solver/gd.h"
#include "solver/cgd.h"
#include "solver/lbfgs.h"
#include "solver/quasi.h"

#include "lsearch/init.h"
#include "lsearch/backtrack.h"
#include "lsearch/cgdescent.h"
#include "lsearch/lemarechal.h"
#include "lsearch/morethuente.h"
#include "lsearch/nocedalwright.h"

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

lsearch_init_factory_t& nano::get_lsearch_inits()
{
    static lsearch_init_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<lsearch_unit_init_t>("unit", "unit line-search step length");
        manager.add<lsearch_linear_init_t>("linear", "linearly interpolate the previous line-search step");
        manager.add<lsearch_quadratic_init_t>("quadratic", "quadratically interpolate the previous line-search step");
        manager.add<lsearch_cgdescent_init_t>("cgdescent", "the initial line-search step length described in CG-DESCENT");
    });

    return manager;
}

lsearch_strategy_factory_t& nano::get_lsearch_strategies()
{
    static lsearch_strategy_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<lsearch_backtrack_t>("backtrack", "backtracking using Armijo conditions");
        manager.add<lsearch_cgdescent_t>("cgdescent", "CG-DESCENT using strong Wolfe conditions");
        manager.add<lsearch_lemarechal_t>("lemarechal", "LeMarechal using regular Wolfe conditions");
        manager.add<lsearch_morethuente_t>("morethuente", "More&Thuente using strong Wolfe conditions");
        manager.add<lsearch_nocedalwright_t>("nocedalwright", "Nocedal&Wright using strong Wolfe conditions");
    });

    return manager;
}

static std::unique_ptr<lsearch_init_t> make_initializer(const lsearch_t::initializer initializer)
{
    switch (initializer)
    {
    case lsearch_t::initializer::unit:          return std::make_unique<lsearch_unit_init_t>();
    case lsearch_t::initializer::linear:        return std::make_unique<lsearch_linear_init_t>();
    case lsearch_t::initializer::quadratic:     return std::make_unique<lsearch_quadratic_init_t>();
    case lsearch_t::initializer::cgdescent:     return std::make_unique<lsearch_cgdescent_init_t>();
    default:                                    assert(false); return nullptr;
    }
}

static std::unique_ptr<lsearch_strategy_t> make_strategy(const lsearch_t::strategy strategy)
{
    switch (strategy)
    {
    case lsearch_t::strategy::backtrack:        return std::make_unique<lsearch_backtrack_t>();
    case lsearch_t::strategy::cgdescent:        return std::make_unique<lsearch_cgdescent_t>();
    case lsearch_t::strategy::lemarechal:       return std::make_unique<lsearch_lemarechal_t>();
    case lsearch_t::strategy::morethuente:      return std::make_unique<lsearch_morethuente_t>();
    case lsearch_t::strategy::nocedalwright:    return std::make_unique<lsearch_nocedalwright_t>();
    default:                                    assert(false); return nullptr;
    }
}

lsearch_t::lsearch_t(const initializer init, const strategy strat, const scalar_t c1, const scalar_t c2) :
    m_initializer(make_initializer(init)),
    m_strategy(make_strategy(strat))
{
    m_strategy->c1(c1);
    m_strategy->c2(c2);
    m_strategy->max_iterations(40);
}

bool lsearch_t::operator()(solver_state_t& state)
{
    // check descent direction
    if (!state.has_descent())
    {
        return false;
    }

    // check parameters
    if (!(  0 < m_strategy->c1() &&
        m_strategy->c1() < scalar_t(0.5) &&
        m_strategy->c1() < m_strategy->c2() &&
        m_strategy->c2() < 1))
    {
        return false;
    }

    // initial step length
    // fixme: is it safe to allow t0 to be greater than 1?!
    const auto t0 = nano::clamp(m_initializer->get(state), lsearch_strategy_t::stpmin(), scalar_t(1));

    // line-search step length
    auto state0 = state;
    state0.t = 0;
    return m_strategy->get(state0, t0, state) && state && (state < state0);
}
