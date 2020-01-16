#include <mutex>
#include <nano/tune.h>
#include <nano/solver/stochastic.h>

using namespace nano;

solver_state_t stochastic_solver_t::init_state(const function_t& function)
{
    solver_state_t state;
    state.x.resize(function.size());
    state.g.resize(function.size());
    state.d.resize(function.size());
    return state;
}

lrate_schedule_t stochastic_solver_t::tune(const solver_function_t& function, vector_t x0, solver_state_t& state) const
{
    auto& x = state.d;
    auto& g = state.g;

    const auto batch = static_cast<tensor_size_t>(batch0());
    const auto tune_begin = tensor_size_t{0};
    const auto tune_end = std::min(function.summands(), static_cast<tensor_size_t>(batch * tuneit()));

    auto evaluate = [&] (const scalar_t lrate)
    {
        auto lrate_schedule = lrate_schedule_t{lrate, decay()};

        x = x0;
        for (tensor_size_t begin = tune_begin; begin + batch <= tune_end; begin += batch, ++ lrate_schedule)
        {
            function.vgrad(x, begin, begin + batch, &g);
            x -= lrate_schedule.get() * g;
        };

        state.f = function.vgrad(x, tune_begin, tune_end);
        state.lrate = lrate;
        state.decay = decay();
        state.g.setConstant(1 + std::fabs(state.f));
        solver_t::done(function, state, true);
        return state.f;
    };

    const auto optimum = geom_tune(evaluate, 1.0, 2.0, 2.0);

    // NB: restore optimum learning rate and decay factor
    state.x = x0;
    state.lrate = std::get<1>(optimum);
    state.decay = decay();

    return {state.lrate, state.decay};
}

solver_sgd_t::solver_sgd_t()
{
    decay(0.75);
}

solver_state_t solver_sgd_t::minimize(const function_t& f, const vector_t& x0) const
{
    const auto function = solver_function_t{f};

    auto state = init_state(function);
    auto lrate_schedule = tune(function, x0, state);
    auto batch_schedule = batch_schedule_t{batch0(), batchr(), function};

    for (int64_t i = 0; i < max_iterations(); ++ i)
    {
        // stochastic updates within an epoch
        batch_schedule.loop(lrate_schedule, [&] (const tensor_size_t begin, const tensor_size_t end, const scalar_t lrate_k)
        {
            function.vgrad(state.x, begin, end, &state.g);
            state.x -= lrate_k * state.g;
            return static_cast<bool>(state);
        });

        // evaluate on all samples at the end of an epoch
        state.f = function.vgrad(state.x, &state.g);
        if (solver_t::done(function, state, true))
        {
            break;
        }
    }

    return state;
}

solver_asgd_t::solver_asgd_t()
{
    decay(0.50);
}

solver_state_t solver_asgd_t::minimize(const function_t& f, const vector_t& x0) const
{
    const auto function = solver_function_t{f};

    auto state = init_state(function);
    auto lrate_schedule = tune(function, x0, state);
    auto batch_schedule = batch_schedule_t{batch0(), batchr(), function};

    auto& xavg = state.d;
    xavg.setZero();

    for (int64_t i = 0; i < max_iterations(); ++ i)
    {
        // stochastic updates within an epoch
        batch_schedule.loop(lrate_schedule, [&] (const tensor_size_t begin, const tensor_size_t end, const scalar_t lrate_k)
        {
            function.vgrad(state.x, begin, end, &state.g);
            state.x -= lrate_k * state.g;
            xavg.noalias() = xavg + (state.x - xavg) / (lrate_schedule.k() + 1);
            return static_cast<bool>(state);
        });

        // evaluate on all samples at the end of an epoch, on the average point
        std::swap(state.x, xavg);

        state.f = function.vgrad(state.x, &state.g);
        if (solver_t::done(function, state, true))
        {
            break;
        }

        std::swap(state.x, xavg);
    }

    return state;
}

stochastic_solver_factory_t& stochastic_solver_t::all()
{
    static stochastic_solver_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<solver_sgd_t>("sgd", "stochastic gradient (descent)");
        manager.add<solver_asgd_t>("asgd", "stochastic gradient (descent) with averaging");
    });

    return manager;
}
