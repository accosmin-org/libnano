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
    // FIXME: configurable solver
    register_parameter(parameter_t::make_scalar("solver::penalty::c0", 0.0, LT, 1.0, LE, 1e+3));
    register_parameter(parameter_t::make_scalar("solver::penalty::gamma", 1.0, LT, 2.0, LE, 1e+3));
}

template <typename tpenalty>
solver_state_t solver_penalty_t<tpenalty>::minimize(const solver_t& solver, const function_t& function,
                                                    const vector_t& x0) const
{
    const auto t0         = 1.0;
    const auto gamma      = 10.0;
    const auto max_outers = tensor_size_t{30};

    auto t                = t0;
    auto x                = x0;
    auto penalties        = vector_t{};
    auto bstate           = solver_state_t{};
    auto penalty_function = tpenalty{function};

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        penalty_function.penalty_term(t);

        // solver->parameter("solver::epsilon") = epsilon;

        const auto state = solver.minimize(penalty_function, x);
        update_penalties(function, state.x, penalties);

        std::cout << std::fixed << std::setprecision(10) << "outer=" << outer << "|" << max_outers << ",t=" << t
                  << ",fx=" << state.f << ",x=" << state.x.transpose() << ",fcalls=" << state.m_fcalls
                  << ",gcalls=" << state.m_gcalls << ",pmin=" << penalties.minCoeff() << ",psum=" << penalties.sum()
                  << ",status=" << state.m_status << std::endl;

        x = state.x;
        t *= gamma;
        bstate = state;
    }

    return bstate;
}

template class nano::solver_penalty_t<linear_penalty_function_t>;
template class nano::solver_penalty_t<quadratic_penalty_function_t>;
