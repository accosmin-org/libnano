#include <nano/solver/penalty.h>

#include <iomanip>
#include <iostream>

using namespace nano;

static void update_penalties(const function_t& function, const vector_t& x, vector_t& penalties)
{
    const auto& constraints = function.constraints();

    penalties.resize(static_cast<tensor_size_t>(constraints.size()));
    for (size_t i = 0U, size = constraints.size(); i < size; ++i)
    {
        penalties(static_cast<tensor_size_t>(i)) = ::nano::valid(constraints[i], x);
    }
}

static auto converged(const function_t& function, const solver_state_t& state, solver_state_t& best_state,
                      vector_t& penalties, scalar_t epsilon)
{
    const auto old_penalties_sum = penalties.sum();
    update_penalties(function, state.x, penalties);
    const auto new_penalties_sum = penalties.sum();

    best_state.m_fcalls += state.m_fcalls;
    best_state.m_gcalls += state.m_gcalls;

    auto converged = false;
    if (new_penalties_sum < old_penalties_sum)
    {
        best_state.f = state.f;
        best_state.x = state.x;

        if (new_penalties_sum < epsilon)
        {
            best_state.m_status = solver_state_t::status::converged;
            // TODO: store penalties
            converged = true;
        }
    }

    return converged;
}

static auto initial_params(const estimator_t& estimator)
{
    const auto eta1       = estimator.parameter("solver::penalty::eta1").template value<scalar_t>();
    const auto eta2       = estimator.parameter("solver::penalty::eta2").template value<scalar_t>();
    const auto cutoff     = estimator.parameter("solver::penalty::cutoff").template value<scalar_t>();
    const auto epsilon    = estimator.parameter("solver::penalty::epsilon").template value<scalar_t>();
    const auto penalty0   = estimator.parameter("solver::penalty::penalty0").template value<scalar_t>();
    const auto max_outers = estimator.parameter("solver::penalty::max_outer_iters").template value<tensor_size_t>();

    return std::make_tuple(eta1, eta2, cutoff, epsilon, penalty0, max_outers);
}

template <typename tpenalty>
static auto make_penalty_function(const function_t& function, const scalar_t cutoff)
{
    auto penalty_function = tpenalty{function};
    penalty_function.cutoff(cutoff);
    return penalty_function;
}

solver_penalty_t::solver_penalty_t()
{
    register_parameter(parameter_t::make_scalar("solver::penalty::eta1", 0.0, LT, 0.1, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::penalty::eta2", 1.0, LT, 20.0, LE, 1e+3));
    register_parameter(parameter_t::make_scalar("solver::penalty::cutoff", 0, LT, 1e+3, LE, 1e+10));
    register_parameter(parameter_t::make_scalar("solver::penalty::epsilon", 0, LT, 1e-8, LE, 1e-1));
    register_parameter(parameter_t::make_scalar("solver::penalty::penalty0", 0.0, LT, 1.0, LE, 1e+3));
    register_parameter(parameter_t::make_integer("solver::penalty::max_outer_iters", 10, LE, 20, LE, 100));
}

solver_state_t solver_linear_penalty_t::minimize(const solver_t& solver, const function_t& function,
                                                 const vector_t& x0) const
{
    [[maybe_unused]] const auto [eta1, eta2, cutoff, epsilon, penalty0, max_outers] = initial_params(*this);

    auto penalty_function = make_penalty_function<linear_penalty_function_t>(function, cutoff);

    auto penalty   = penalty0;
    auto penalties = vector_t{};

    auto best_state = solver_state_t{function, x0};

    update_penalties(function, best_state.x, penalties);

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        penalty_function.penalty(penalty);
        const auto state = solver.minimize(penalty_function, best_state.x);

        const auto converged = ::converged(function, state, best_state, penalties, epsilon);

        std::cout << std::fixed << std::setprecision(10) << "o=" << outer << "|" << max_outers << ",c=" << penalty
                  << ",p=" << penalties.sum() << "," << state << ",x=" << state.x.transpose() << std::endl;

        if (converged)
        {
            break;
        }
        penalty *= eta2;
    }

    return best_state;
}

solver_state_t solver_quadratic_penalty_t::minimize(const solver_t& solver, const function_t& function,
                                                    const vector_t& x0) const
{
    [[maybe_unused]] const auto [eta1, eta2, cutoff, epsilon, penalty0, max_outers] = initial_params(*this);

    auto penalty_function = make_penalty_function<quadratic_penalty_function_t>(function, cutoff);

    auto penalty = penalty0;
    auto penalties = vector_t{};

    auto best_state = solver_state_t{function, x0};

    update_penalties(function, best_state.x, penalties);

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        penalty_function.penalty(penalty);
        const auto state = solver.minimize(penalty_function, best_state.x);

        const auto converged = ::converged(function, state, best_state, penalties, epsilon);

        std::cout << std::fixed << std::setprecision(10) << "o=" << outer << "|" << max_outers << ",c=" << penalty
                  << ",p=" << penalties.sum() << "," << state << ",x=" << state.x.transpose() << std::endl;

        if (converged)
        {
            break;
        }
        penalty *= eta2;
    }

    return best_state;
}

solver_state_t solver_linear_quadratic_penalty_t::minimize(const solver_t& solver, const function_t& function,
                                                           const vector_t& x0) const
{
    [[maybe_unused]] const auto [eta1, eta2, cutoff, epsilon, penalty0, max_outers] = initial_params(*this);

    auto penalty_function = make_penalty_function<linear_quadratic_penalty_function_t>(function, cutoff);

    auto penalty   = penalty0;
    auto penalties = vector_t{};

    auto best_state = solver_state_t{function, x0};
    update_penalties(function, best_state.x, penalties);

    auto smoothing = penalties.maxCoeff();

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        penalty_function.penalty(penalty);
        penalty_function.smoothing(smoothing);
        const auto state = solver.minimize(penalty_function, best_state.x);

        const auto converged = ::converged(function, state, best_state, penalties, epsilon);

        std::cout << std::fixed << std::setprecision(10) << "o=" << outer << "|" << max_outers << ",c=" << penalty
                  << ",p=" << penalties.sum() << ",e=" << smoothing << "," << state << ",x=" << state.x.transpose()
                  << std::endl;

        if (converged)
        {
            break;
        }

        if (penalties.maxCoeff() <= smoothing)
        {
            if (penalties.sum() < epsilon)
            {
                best_state.m_status = solver_state_t::status::converged;
                break;
            }
            else
            {
                smoothing = eta1 * penalties.maxCoeff();
            }
        }
        else
        {
            penalty *= eta2;
        }
    }

    return best_state;
}
