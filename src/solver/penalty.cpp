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

template <typename tpenalty>
solver_penalty_t<tpenalty>::solver_penalty_t()
{
    register_parameter(parameter_t::make_scalar("solver::penalty::t0", 0.0, LT, 1.0, LE, 1e+3));
    register_parameter(parameter_t::make_scalar("solver::penalty::gamma", 1.0, LT, 10.0, LE, 1e+3));
    register_parameter(parameter_t::make_scalar("solver::penalty::epsilon", 0, LT, 1e-8, LE, 1e-1));
    register_parameter(parameter_t::make_integer("solver::penalty::max_outer_iters", 10, LE, 20, LE, 100));
}

template <typename tpenalty>
solver_state_t solver_penalty_t<tpenalty>::minimize(const solver_t& solver, const function_t& function,
                                                    const vector_t& x0) const
{
    const auto t0         = parameter("solver::penalty::t0").template value<scalar_t>();
    const auto gamma      = parameter("solver::penalty::gamma").template value<scalar_t>();
    const auto epsilon    = parameter("solver::penalty::epsilon").template value<scalar_t>();
    const auto max_outers = parameter("solver::penalty::max_outer_iters").template value<tensor_size_t>();

    auto penalty_term     = t0;
    auto penalty_function = tpenalty{function};
    auto best_state       = solver_state_t{function, x0};

    auto penalties = vector_t{};
    update_penalties(function, best_state.x, penalties);

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        penalty_function.penalty_term(penalty_term);
        const auto state = solver.minimize(penalty_function, best_state.x);

        const auto old_penalties_sum = penalties.sum();
        update_penalties(function, state.x, penalties);
        const auto new_penalties_sum = penalties.sum();

        std::cout << std::fixed << std::setprecision(10) << "o=" << outer << "|" << max_outers << ",t=" << penalty_term
                  << ",p=" << new_penalties_sum << "," << state << std::endl;

        penalty_term *= gamma;
        best_state.m_fcalls += state.m_fcalls;
        best_state.m_gcalls += state.m_gcalls;

        if (new_penalties_sum < old_penalties_sum)
        {
            best_state.f = state.f;
            best_state.x = state.x;

            if (penalties.sum() < epsilon)
            {
                best_state.m_status = solver_state_t::status::converged;

                // TODO: store the total number of function evaluations...
                // TODO: store penalties
                break;
            }
        }
    }

    return best_state;
}

template class nano::solver_penalty_t<linear_penalty_function_t>;
template class nano::solver_penalty_t<quadratic_penalty_function_t>;
