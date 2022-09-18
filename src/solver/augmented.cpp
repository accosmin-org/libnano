#include <nano/function/penalty.h>
#include <nano/solver/augmented.h>

using namespace nano;

solver_augmented_lagrangian_t::solver_augmented_lagrangian_t()
    : solver_penalty_t("augmented-lagrangian")
{
    type(solver_type::constrained);

    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();
    static constexpr auto fmin = std::numeric_limits<scalar_t>::lowest();

    register_parameter(parameter_t::make_scalar("solver::augmented::tau", 0.0, LT, 0.5, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::augmented::gamma", 1.0, LT, 10.0, LT, fmax));
    register_parameter(parameter_t::make_scalar("solver::augmented::miu_max", 0.0, LT, +1e+20, LT, fmax));
    register_parameter(parameter_t::make_integer("solver::augmented::max_outer_iters"), 10, LE, 20, LE, 100);
    register_parameter(parameter_t::make_scalar_pair("solver::augmented::lambda", fmin, LT, -1e+20, LT, +1e+20, fmax));
}

rsolver_t solver_augmented_lagrangian_t::clone() const
{
    return std::make_unique<solver_augmented_lagrangian_t>(*this);
}

solver_state_t solver_augmented_lagrangian_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    const auto epsilon                  = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals                = parameter("solver::max_evals").value<tensor_size_t>();
    const auto tau                      = parameter("solver::augmented::tau").value<scalar_t>();
    const auto gamma                    = parameter("solver::augmented::gamma").value<scalar_t>();
    const auto miu_max                  = parameter("solver::augmented::miu_max").value<scalar_t>();
    const auto [lambda_min, lambda_max] = parameter("solver::augmented::lambda").value_pair<scalar_t>();
    const auto max_outers               = parameter("solver::augmented::max_outer_iters").value<tensor_size_t>();

    vector_t lambda = vector_t::Zero(::nano::count_equalities(function));
    vector_t miu    = vector_t::Zero(::nano::count_inequalities(function));

    auto bstate = solver_state_t{function, x0};
    auto solver = make_solver(penalty_function, epsilon, max_evals);
    auto augmented_function = augmented_function_t{function, lambda, miu};

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        penalty_function.penalty(penalty);

        const auto cstate    = solver->minimize(penalty_function, bstate.x);
        const auto iter_ok   = cstate.valid();
        const auto converged = iter_ok && ::converged(cstate, bstate, epsilon);

        if (done(penalty_function.function(), bstate, iter_ok, converged))
        {
            break;
        }

        penalty *= eta;
    }

    return bstate;
}
