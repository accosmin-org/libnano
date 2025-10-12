#include <Eigen/Dense>
#include <solver/newton.h>

using namespace nano;

solver_newton_t::solver_newton_t()
    : solver_t("newton")
{
    parameter("solver::tolerance") = std::make_tuple(1e-1, 9e-1);
}

rsolver_t solver_newton_t::clone() const
{
    return std::make_unique<solver_newton_t>(*this);
}

bool solver_newton_t::has_lsearch() const
{
    return true;
}

solver_state_t solver_newton_t::do_minimize(const function_t& function, const vector_t& x0,
                                            const logger_t& logger) const
{
    solver_t::warn_nonconvex(function, logger);
    solver_t::warn_nonsmooth(function, logger);
    solver_t::warn_constrained(function, logger);

    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();

    // TODO: variant with computing the hessian's inverse every X(=5) iterations

    auto cstate = solver_state_t{function, x0};
    if (cstate.gx().lpNorm<Eigen::Infinity>() < epsilon0<scalar_t>())
    {
        solver_t::done_gradient_test(cstate, cstate.valid(), logger);
        return cstate;
    }

    auto pstate  = cstate;
    auto lsearch = make_lsearch();
    auto descent = vector_t{function.size()};
    auto hessian = matrix_t{function.size(), function.size()};
    auto solver  = Eigen::LDLT<eigen_matrix_t<scalar_t>>{};

    while (function.fcalls() + function.gcalls() + function.hcalls() < max_evals)
    {
        // descent direction
        function(cstate.x(), {}, hessian);

        solver.compute(hessian.matrix());
        descent.vector() = solver.solve(-cstate.gx());

        // line-search
        pstate             = cstate;
        const auto iter_ok = lsearch.get(cstate, descent, logger);
        if (solver_t::done_gradient_test(cstate, iter_ok, logger))
        {
            break;
        }
    }

    return cstate.valid() ? cstate : pstate;
} // LCOV_EXCL_LINE
