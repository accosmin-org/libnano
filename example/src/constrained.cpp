#include <iomanip>
#include <iostream>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/lambda.h>
#include <nano/solver.h>

using namespace nano;

int main(const int, char*[])
{
    // construct the constrained problem:
    //  min f(x) = 1/2 * x.dot(P * x) + q.dot(x) + r
    //  s.t -1 <= x_i <= 1, i=1..3
    //
    // with solution: (1.0, 0.5, -1.0)
    //
    // see exercise 4.3, "Convex optimization", Boyd & Vanderberghe
    const auto xbest  = make_vector<scalar_t>(1.0, 0.5, -1.0);
    const auto lambda = [](vector_cmap_t x, vector_map_t gx)
    {
        static const auto P = make_matrix<scalar_t>(3, 13, 12, -2, 12, 17, 6, -2, 6, 12);
        static const auto q = make_vector<scalar_t>(-22, -14.5, 13.0);
        static const auto r = 1.0;
        if (gx.size() == x.size())
        {
            gx = P * x + q;
        }
        return 0.5 * x.dot(P * x) + x.dot(q) + r;
    };

    auto function = make_function(3, convexity::yes, smoothness::yes, 0.0, lambda);
    critical(function.optimum(xbest));
    critical(function.variable() <= +1.0);
    critical(function.variable() >= -1.0);

    // construct an appropriate solver_t object to solve the constrained problem
    // NB: this may be not needed as the default configuration will minimize the objective as well!
    auto solver = solver_t::all().get("augmented-lagrangian");
    assert(solver != nullptr);
    solver->parameter("solver::augmented::base_solver_id") = "lbfgs";
    solver->parameter("solver::epsilon")                   = 1e-7;
    solver->parameter("solver::max_evals")                 = 50000;

    // minimize starting from various random points
    for (auto trial = 0, trials = 10; trial < trials; ++trial)
    {
        const auto x0 = make_random_vector<scalar_t>(function.size());
        std::cout << std::fixed << std::setprecision(12) << "minimize[" << (trial + 1) << "/" << trials
                  << "]: f0=" << function(x0) << "...\n";

        // minimize
        const auto state = solver->minimize(function, x0, make_stdout_logger());
        const auto error = (state.x() - xbest).lpNorm<Eigen::Infinity>();

        assert(state.status() == solver_status::kkt_optimality_test);

        std::cout << std::fixed << std::setprecision(12) << "minimize[" << (trial + 1) << "/" << trials
                  << "]: f0=" << function(x0) << ",x-x*=" << error << "," << state << ".\n";

        if (error > 1e-7)
        {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
