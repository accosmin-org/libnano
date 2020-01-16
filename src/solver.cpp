#include <mutex>
#include <nano/solver/gd.h>
#include <nano/solver/cgd.h>
#include <nano/solver/lbfgs.h>
#include <nano/solver/quasi.h>
#include <nano/solver/stochastic.h>

using namespace nano;

void solver_t::logger(const logger_t& logger)
{
    m_logger = logger;
}

void solver_t::epsilon(const scalar_t epsilon)
{
    m_epsilon = epsilon;
}

void solver_t::max_iterations(const int max_iterations)
{
    m_max_iterations = max_iterations;
}

bool solver_t::done(const solver_function_t& function, solver_state_t& state, const bool iter_ok) const
{
    state.m_fcalls = function.fcalls();
    state.m_gcalls = function.gcalls();

    const auto step_ok = iter_ok && state;
    const auto converged = state.converged(epsilon());

    if (converged || !step_ok)
    {
        // either converged or failed
        state.m_status = converged ?
            solver_state_t::status::converged :
            solver_state_t::status::failed;
        log(state);
        return true;
    }
    else if (!log(state))
    {
        // stopping was requested
        state.m_status = solver_state_t::status::stopped;
        return true;
    }

    // OK, go on with the optimization
    return false;
}

bool solver_t::log(solver_state_t& state) const
{
    const auto status = !m_logger ? true : m_logger(state);
    state.m_iterations ++;
    return status;
}

solver_factory_t& solver_t::all()
{
    static solver_factory_t manager;

    // FIXME: use the solvers registered to lsearch_solver_t and stochastic_solver_t!!!

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<solver_gd_t>("gd", "gradient descent");
        manager.add<solver_sgd_t>("sgd", "stochastic gradient (descent)");
        manager.add<solver_asgd_t>("asgd", "stochastic gradient (descent) with averaging");
        manager.add<solver_cgd_pr_t>("cgd", "conjugate gradient descent (default)");
        manager.add<solver_cgd_n_t>("cgd-n", "conjugate gradient descent (N+)");
        manager.add<solver_cgd_hs_t>("cgd-hs", "conjugate gradient descent (HS+)");
        manager.add<solver_cgd_fr_t>("cgd-fr", "conjugate gradient descent (FR)");
        manager.add<solver_cgd_pr_t>("cgd-pr", "conjugate gradient descent (PR+)");
        manager.add<solver_cgd_cd_t>("cgd-cd", "conjugate gradient descent (CD)");
        manager.add<solver_cgd_ls_t>("cgd-ls", "conjugate gradient descent (LS+)");
        manager.add<solver_cgd_dy_t>("cgd-dy", "conjugate gradient descent (DY)");
        manager.add<solver_cgd_dycd_t>("cgd-dycd", "conjugate gradient descent (DYCD)");
        manager.add<solver_cgd_dyhs_t>("cgd-dyhs", "conjugate gradient descent (DYHS)");
        manager.add<solver_cgd_frpr_t>("cgd-prfr", "conjugate gradient descent (FRPR)");
        manager.add<solver_lbfgs_t>("lbfgs", "limited-memory BFGS");
        manager.add<solver_quasi_dfp_t>("dfp", "quasi-newton method (DFP)");
        manager.add<solver_quasi_sr1_t>("sr1", "quasi-newton method (SR1)");
        manager.add<solver_quasi_bfgs_t>("bfgs", "quasi-newton method (BFGS)");
        manager.add<solver_quasi_hoshino_t>("hoshino", "quasi-newton method (Hoshino formula)");
        manager.add<solver_quasi_fletcher_t>("fletcher", "quasi-newton method (Fletcher's switch)");
    });

    return manager;
}
