#include <nano/solver.h>
#include <nano/model/tuner.h>
#include <nano/core/logger.h>
#include <nano/model/surrogate.h>

using namespace nano;

template <typename ttensor, typename tfunc>
static tensor1d_t map(const param_spaces_t& spaces, const ttensor& values, const tfunc& func)
{
    tensor1d_t result(values.size());
    for (tensor_size_t i = 0, size = values.size(); i < size; ++ i)
    {
        result(i) = func(spaces[static_cast<size_t>(i)], values(i));
    }
    return result;
}

static void update(std::vector<tuner_t::step_t>& steps)
{
    if (steps.size() > 1U)
    {
        auto& current = steps[steps.size() - 1U];
        const auto& previous = steps[steps.size() - 2U];

        if (previous.m_opt_value < current.m_value)
        {
            current.m_opt_param = previous.m_opt_param;
            current.m_opt_value = previous.m_opt_value;
        }
        else
        {
            current.m_opt_param = current.m_param;
            current.m_opt_value = current.m_value;
        }
    }
}

tuner_t::step_t::step_t() = default;

tuner_t::step_t::step_t(
    tensor1d_t param, solver_state_t surrogate_fit, solver_state_t surrogate_opt, const callback_t& callback) :
    m_param(std::move(param)),
    m_opt_param(m_param),
    m_value(callback(m_param)),
    m_opt_value(m_value),
    m_surrogate_fit(std::move(surrogate_fit)),
    m_surrogate_opt(std::move(surrogate_opt))
{
    critical(
        !std::isfinite(m_value),
        "tuner: not finite value (", m_value, ") computed for parameters (", m_param.vector().transpose(), ")!");

    m_surrogate_fit.function = nullptr;
    m_surrogate_opt.function = nullptr;
}

tuner_t::tuner_t(param_spaces_t param_spaces, tuner_t::callback_t callback) :
    m_param_spaces(std::move(param_spaces)),
    m_callback(std::move(callback))
{
    critical(
        m_param_spaces.empty(),
        "tuner: at least a parameter space is needed!");

    register_parameter(parameter_t::make_integer("tuner::max_iterations", 0, LE, 5, LE, 100));
    register_parameter(parameter_t::make_scalar("tuner::solver::max_evals", 100, LE, 1000, LE, 100'000));
    register_parameter(parameter_t::make_scalar("tuner::solver::epsilon", 1e-16, LE, 1e-8, LE, 1e-6));
}

tuner_t::steps_t tuner_t::optimize(const tensor2d_t& initial_params) const
{
    const auto psize = m_param_spaces.size();
    const auto min_history = (psize + 1) * (psize + 2) / 2U;
    const auto max_iterations = parameter("tuner::max_iterations").value<tensor_size_t>();

    critical(
        initial_params.size<1>() != static_cast<tensor_size_t>(psize),
        "tuner: received ", initial_params.size<1>(), " parameters, expecting ", psize, "!");

    const auto solver = solver_t::all().get("lbfgs");
    assert(solver != nullptr);
    solver->parameter("solver::epsilon") = parameter("tuner::solver::epsilon").value<scalar_t>();
    solver->parameter("solver::max_evals") = parameter("tuner::solver::max_evals").value<scalar_t>();

    const auto loss = loss_t::all().get("squared");
    assert(loss != nullptr);

    const auto to_surrogate = [] (const auto& space, auto value)
    {
        return space.to_surrogate(value);
    };
    const auto from_surrogate = [] (const auto& space, auto value)
    {
        return space.closest_grid_value_from_surrogate(value);
    };

    steps_t steps;
    for (tensor_size_t sample = 0; sample < initial_params.size<0>(); ++ sample)
    {
        steps.emplace_back(
            initial_params.tensor(sample),
            solver_state_t{},
            solver_state_t{},
            m_callback);

        ::update(steps);
    }

    for (tensor_size_t iteration = 0; iteration < max_iterations && steps.size() >= min_history; ++ iteration)
    {
        tensor2d_t p(static_cast<tensor_size_t>(steps.size()), static_cast<tensor_size_t>(m_param_spaces.size()));
        tensor1d_t y(static_cast<tensor_size_t>(steps.size()));

        tensor_size_t k = 0;
        for (const auto& step : steps)
        {
            p.tensor(k) = map(m_param_spaces, step.m_param, to_surrogate);
            y(k ++) = step.m_value;
        }

        const auto surrogate_fit = quadratic_surrogate_fit_t{*loss, p, y};
        auto state_fit = solver->minimize(surrogate_fit, vector_t::Zero(surrogate_fit.size()));

        const auto surrogate_opt = quadratic_surrogate_t{state_fit.x};
        auto state_opt = solver->minimize(surrogate_opt,
            map(m_param_spaces, steps.rbegin()->m_opt_param.vector(), to_surrogate).vector());

        auto param = map(m_param_spaces, state_opt.x, from_surrogate);
        const auto itdup = find_if(steps.begin(), steps.end(), [&] (const auto& step)
        {
            static const auto epsilon = epsilon0<scalar_t>();
            return (param.vector() - step.m_param.vector()).template lpNorm<Eigen::Infinity>() < epsilon;
        });
        if (itdup != steps.end())
        {
            break;
        }

        steps.emplace_back(
            std::move(param),
            std::move(state_fit),
            std::move(state_opt),
            m_callback);

        ::update(steps);
    }

    return steps;
}
