#include <iomanip>
#include <iostream>
#include <nano/solver/lbfgs.h>

using namespace nano;

class objective_t final : public function_t
{
public:

    objective_t(const int size) :
        function_t("objective's name", size),
        m_b(vector_t::Random(size))
    {
        convex(true);
        smooth(true);
    }

    scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr, vgrad_config_t = vgrad_config_t{}) const override
    {
        assert(size() == x.size());
        assert(size() == m_b.size());

        const auto dx = 1 + (x - m_b).dot(x - m_b) / 2;

        if (gx != nullptr)
        {
            *gx = (x - m_b) / dx;
        }

        return std::log(dx);
    }

    const auto& b() const { return m_b; }

private:

    // attributes
    vector_t    m_b;
};

int main(const int, char* [])
{
    // construct an objective function
    const auto objective = objective_t{13};

    // check the objective function's gradient using central finite-difference
    const auto trials = 10;
    for (auto trial = 0; trial < 10; ++ trial)
    {
        const vector_t x0 = nano::vector_t::Random(objective.size());

        std::cout << std::fixed << std::setprecision(12)
            << "check_grad[" << (trial + 1) << "/" << trials
            << "]: dg=" << objective.grad_accuracy(x0) << std::endl;
    }
    std::cout << std::endl;

    // construct a solver_t object to minimize the objective function
    // NB: this may be not needed as the default configuration will minimize the objective as well!
    // NB: can also use the factory to get a default solver!
    auto solver = nano::solver_lbfgs_t{};
    solver.parameter("solver::lbfgs::history") = 6;
    solver.parameter("solver::epsilon") = 1e-6;
    solver.parameter("solver::max_evals") = 100;
    solver.parameter("solver::tolerance") = std::make_tuple(1e-4, 9e-1);
    solver.lsearch0("constant");
    solver.lsearchk("morethuente");

    // minimize starting from various random points
    for (auto trial = 0; trial < trials; ++ trial)
    {
        const vector_t x0 = nano::vector_t::Random(objective.size());

        std::cout << std::fixed << std::setprecision(12)
            << "minimize[" << (trial + 1) << "/" << trials
            << "]: f0=" << objective.vgrad(x0, nullptr)
            << "...\n";

        // log the optimization steps
        solver.logger([&] (const nano::solver_state_t& state)
        {
            std::cout
                << "\tdescent: i=" << state.m_iterations
                << ",f=" << state.f << ",g=" << state.convergence_criterion()
                << "[" << state.m_status << "]" << ",calls=" << state.m_fcalls << "/" << state.m_gcalls
                << ".\n";
            return true;
        });

        // log the line-search steps
        solver.lsearch0_logger([&] (const nano::solver_state_t& state0, const nano::scalar_t t0)
        {
            std::cout
                << "\t\tlsearch(0): t=" << state0.t << ",f=" << state0.f << ",g=" << state0.convergence_criterion()
                << ",t0=" << t0 <<".\n";
        });

        const auto [c1, c2] = solver.parameter("solver::tolerance").value_pair<scalar_t>();

        solver.lsearchk_logger([&, c1=c1, c2=c2] (const nano::solver_state_t& state0, const nano::solver_state_t& state)
        {
            std::cout
                << "\t\tlsearch(t): t=" << state.t << ",f=" << state.f << ",g=" << state.convergence_criterion()
                << ",armijo=" << state.has_armijo(state0, c1)
                << ",wolfe=" << state.has_wolfe(state0, c2)
                << ",swolfe=" << state.has_strong_wolfe(state0, c2) << ".\n";
        });

        const auto state = solver.minimize(objective, x0);

        std::cout << std::fixed << std::setprecision(12)
            << "minimize[" << (trial + 1) << "/" << trials
            << "]: f0=" << objective.vgrad(x0, nullptr)
            << ", f=" << state.f
            << ", g=" << state.convergence_criterion()
            << ", x-x*=" << (state.x - objective.b()).lpNorm<Eigen::Infinity>()
            << ", iters=" << state.m_iterations
            << ", fcalls=" << state.m_fcalls
            << ", gcalls=" << state.m_gcalls
            << ", status=" << state.m_status
            << "\n" << std::endl;
    }

    return EXIT_SUCCESS;
}
