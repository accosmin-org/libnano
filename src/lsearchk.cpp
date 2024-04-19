#include <mutex>
#include <nano/core/numeric.h>
#include <nano/lsearchk/backtrack.h>
#include <nano/lsearchk/cgdescent.h>
#include <nano/lsearchk/fletcher.h>
#include <nano/lsearchk/lemarechal.h>
#include <nano/lsearchk/morethuente.h>

using namespace nano;

lsearchk_t::lsearchk_t(string_t id)
    : clonable_t(std::move(id))
{
    register_parameter(parameter_t::make_scalar_pair("lsearchk::tolerance", 0, LT, 1e-4, LT, 0.1, LT, 1));
    register_parameter(parameter_t::make_integer("lsearchk::max_iterations", 1, LE, 128, LE, 10000));
}

factory_t<lsearchk_t>& lsearchk_t::all()
{
    static auto manager = factory_t<lsearchk_t>{};
    const auto  op      = []()
    {
        manager.add<lsearchk_fletcher_t>("Fletcher (strong Wolfe conditions)");
        manager.add<lsearchk_backtrack_t>("backtrack using cubic interpolation (Armijo conditions)");
        manager.add<lsearchk_cgdescent_t>("CG-DESCENT (regular and approximate Wolfe conditions)");
        manager.add<lsearchk_lemarechal_t>("LeMarechal (regular Wolfe conditions)");
        manager.add<lsearchk_morethuente_t>("More&Thuente (strong Wolfe conditions)");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}

lsearchk_t::result_t lsearchk_t::get(solver_state_t& state, const vector_t& descent, scalar_t step_size) const
{
    const auto max_iterations = parameter("lsearchk::max_iterations").value<int>();

    // check descent direction
    if (!state.has_descent(descent))
    {
        return {false, step_size};
    }

    // adjust the initial step size if it produces an invalid state
    const auto state0 = state;

    step_size = std::isfinite(step_size) ? std::clamp(step_size, stpmin(), 1.0) : scalar_t(1);
    for (int i = 0; i < max_iterations && !update(state, state0, descent, step_size); ++i)
    {
        step_size *= 0.3;
    }

    // adjust the initial step if the function value is too close (e.g. badly conditioned function)
    for (int i = 0; i < max_iterations && std::fabs(state.fx() - state0.fx()) < epsilon1<scalar_t>(); ++i)
    {
        step_size *= 3.0;
        if (!update(state, state0, descent, step_size))
        {
            return {false, step_size};
        }
    }

    // line-search step size
    // NB: some line-search algorithms (see CGDESCENT) allow a small increase
    //     in the function value when close to numerical precision!
    return do_get(state0, descent, step_size, state);
}

void lsearchk_t::logger(const lsearchk_t::logger_t& logger)
{
    m_logger = logger;
}

bool lsearchk_t::update(solver_state_t& state, const solver_state_t& state0, const vector_t& descent,
                        const scalar_t step_size) const
{
    const auto ok = state.update(state0.x() + step_size * descent);
    if (m_logger)
    {
        m_logger(state0, state, descent, step_size);
    }
    return ok;
}

void lsearchk_t::type(const lsearch_type type)
{
    m_type = type;
}

scalar_t lsearchk_t::stpmin()
{
    return scalar_t(10) * std::numeric_limits<scalar_t>::epsilon();
}

scalar_t lsearchk_t::stpmax()
{
    return scalar_t(1) / stpmin();
}

lsearch_type lsearchk_t::type() const
{
    return m_type;
}
