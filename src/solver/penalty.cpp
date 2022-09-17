#include <nano/function/penalty.h>
#include <nano/solver/ellipsoid.h>
#include <nano/solver/lbfgs.h>
#include <nano/solver/penalty.h>

using namespace nano;

static void register_params(const char* const prefix, estimator_t& estimator)
{
    estimator.register_parameter(parameter_t::make_scalar(scat(prefix, "::eta"), 1.0, LT, 10.0, LE, 1e+3));
    estimator.register_parameter(parameter_t::make_scalar(scat(prefix, "::penalty0"), 0.0, LT, 1.0, LE, 1e+3));
    estimator.register_parameter(parameter_t::make_integer(scat(prefix, "::max_outer_iters"), 10, LE, 20, LE, 100));
}

static auto initial_params(const char* const prefix, const estimator_t& estimator)
{
    const auto epsilon    = estimator.parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals  = estimator.parameter("solver::max_evals").value<tensor_size_t>();
    const auto eta        = estimator.parameter(scat(prefix, "::eta")).value<scalar_t>();
    const auto penalty0   = estimator.parameter(scat(prefix, "::penalty0")).value<scalar_t>();
    const auto max_outers = estimator.parameter(scat(prefix, "::max_outer_iters")).value<tensor_size_t>();

    return std::make_tuple(epsilon, max_evals, eta, penalty0, max_outers);
}

template <typename tsolver>
static rsolver_t make_solver(const scalar_t epsilon, const tensor_size_t max_evals, const solver_t::logger_t& logger)
{
    auto solver = std::make_unique<tsolver>();
    solver->logger(logger);
    if (epsilon < 1e-7 && solver->type() == solver_type::line_search)
    {
        // NB: CG-DESCENT line-search gives higher precision than default More&Thuente line-search.
        solver->lsearchk("cgdescent");
    }
    solver->parameter("solver::epsilon")   = epsilon;
    solver->parameter("solver::max_evals") = max_evals;
    return solver;
}

static auto converged(const solver_state_t& curr_state, solver_state_t& best_state, const scalar_t epsilon)
{
    const auto df = std::fabs(curr_state.f - best_state.f);
    const auto dx = (curr_state.x - best_state.x).lpNorm<Eigen::Infinity>();

    const auto pimproved = curr_state.p.sum() <= best_state.p.sum() + epsilon;
    if (pimproved)
    {
        best_state.f = curr_state.f;
        best_state.x = curr_state.x;
        best_state.g = curr_state.g;
        best_state.p = curr_state.p;
    }
    best_state.status = curr_state.status;
    best_state.inner_iters += curr_state.inner_iters;
    best_state.outer_iters++;

    return pimproved && df < epsilon && dx < epsilon;
}

solver_penalty_t::solver_penalty_t(string_t id)
    : solver_t(std::move(id))
{
}

rsolver_t solver_penalty_t::make_solver(const penalty_function_t& penalty_function, const scalar_t epsilon,
                                        const tensor_size_t max_evals) const
{
    // TODO: find a more reliable non-smooth unconstrained solver
    return penalty_function.smooth() ? ::make_solver<solver_lbfgs_t>(epsilon, max_evals, logger())
                                     : ::make_solver<solver_ellipsoid_t>(epsilon, max_evals, logger());
}

solver_state_t solver_penalty_t::minimize(penalty_function_t& penalty_function, const vector_t& x0,
                                          const char* const prefix) const
{
    const auto [epsilon, max_evals, eta, penalty0, max_outers] = initial_params(prefix, *this);

    auto penalty = penalty0;
    auto solver  = make_solver(penalty_function, epsilon, max_evals);
    auto bstate  = solver_state_t{penalty_function.function(), x0};

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        penalty_function.penalty(penalty);

        const auto cstate    = solver->minimize(penalty_function, bstate.x);
        const auto iter_ok   = cstate.valid();
        const auto converged = iter_ok && ::converged(cstate, bstate, epsilon);

        if (cstate.status == solver_status::stopped || done(penalty_function.function(), bstate, iter_ok, converged))
        {
            break;
        }

        penalty *= eta;
    }

    return bstate;
}

solver_linear_penalty_t::solver_linear_penalty_t()
    : solver_penalty_t("linear-penalty")
{
    type(solver_type::constrained);
    register_params("solver::linear_penalty", *this);
}

rsolver_t solver_linear_penalty_t::clone() const
{
    return std::make_unique<solver_linear_penalty_t>(*this);
}

solver_state_t solver_linear_penalty_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    auto penalty_function = linear_penalty_function_t{function};

    return minimize(penalty_function, x0, "solver::linear_penalty");
}

solver_quadratic_penalty_t::solver_quadratic_penalty_t()
    : solver_penalty_t("quadratic-penalty")
{
    type(solver_type::constrained);
    register_params("solver::quadratic_penalty", *this);
}

rsolver_t solver_quadratic_penalty_t::clone() const
{
    return std::make_unique<solver_quadratic_penalty_t>(*this);
}

solver_state_t solver_quadratic_penalty_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    auto penalty_function = quadratic_penalty_function_t{function};

    return minimize(penalty_function, x0, "solver::quadratic_penalty");
}
