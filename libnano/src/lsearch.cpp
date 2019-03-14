#include <mutex>
#include <nano/numeric.h>
#include "lsearch/init.h"
#include "lsearch/backtrack.h"
#include "lsearch/cgdescent.h"
#include "lsearch/lemarechal.h"
#include "lsearch/morethuente.h"
#include "lsearch/nocedalwright.h"

using namespace nano;

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
    return m_strategy->get(state0, t0, state) && state && (state < state0);
}
