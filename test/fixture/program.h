#include <fixture/solver.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>
#include <nano/function/linear.h>
#include <nano/function/quadratic.h>

using namespace nano;

/*inline auto make_permutation(const tensor_size_t m)
{
    auto permutation = arange(0, m);
    std::shuffle(permutation.begin(), permutation.end(), make_rng());
    return permutation;
}

inline auto duplicate(const program::equality_t<matrix_t, vector_t>& equality, const scalar_t dep_w1,
                      const scalar_t dep_w2)
{
    const auto& A = equality.m_A;
    const auto& b = equality.m_b;

    const auto m = A.rows();
    const auto n = A.cols();

    auto b2 = vector_t{2 * m};
    auto A2 = matrix_t{2 * m, n};

    const auto permutation = make_permutation(m);
    for (tensor_size_t row = 0; row < m; ++row)
    {
        const auto permuted_row = permutation(row);
        const auto permuted_mix = (permuted_row + 1) % m;
        const auto duplicat_row = 2 * m - 1 - row;

        b2(row)          = b(permuted_row);
        b2(duplicat_row) = b(permuted_row) * dep_w1 + b(permuted_mix) * dep_w2;

        A2.row(row)          = A.row(permuted_row);
        A2.row(duplicat_row) = A.row(permuted_row).array() * dep_w1 + A.row(permuted_mix).array() * dep_w2;
    }

    return program::make_equality(A2, b2);
}

template <class tprogram>
void check_solution(const tprogram& program, const program::expected_t& expected, const vector_t& x, const scalar_t fx,
                    const scalar_t kkt)
{
    // check optimum (if known)
    if (expected.m_xbest.size() > 0)
    {
        UTEST_CHECK_CLOSE(x, expected.m_xbest, expected.m_epsilon);
    }

    // check optimum function value (if known)
    if (std::isfinite(expected.m_fbest))
    {
        UTEST_CHECK_CLOSE(fx, expected.m_fbest, expected.m_epsilon);
    }

    // check KKT optimality conditions (always verifiable for convex problems)
    UTEST_CHECK(program.feasible(x, expected.m_epsilon));
    if (std::isfinite(kkt))
    {
        UTEST_CHECK_LESS(kkt, expected.m_epsilon);
    }
}*/

static void check_solution_interior(const function_t& function, const logger_t& logger)
{
    UTEST_NAMED_CASE(function.name());

    auto solver = make_solver("ipm");

    const auto x0      = vector_t::constant(function.size(), 1.0);
    const auto state   = solver->minimize(function, x0, logger);
    const auto epsilon = solver->parameter("solver::epsilon").value<scalar_t>();

    const auto& optimum = function.optimum();
    UTEST_CHECK_EQUAL(state.status(), optimum.m_status);
    UTEST_CHECK(optimum.m_xbest.size() == 0 || optimum.m_xbest.size() == state.x().size());

    if (optimum.m_xbest.size() == state.x().size())
    {
        UTEST_CHECK_CLOSE(state.x(), optimum.m_xbest, epsilon);
    }

    if (std::isfinite(optimum.m_fbest))
    {
        UTEST_CHECK_CLOSE(state.fx(), optimum.m_fbest, epsilon);
    }

    if (optimum.m_status == solver_status::converged)
    {
        // FIXME: merge these tests with the generic ones for solver_t
        // FIXME: handle the case of smooth functions (always expected convergence) or with constraints explicitly
        // UTEST_CHECK_LESS(state.gradient_test(), epsilon);
        UTEST_CHECK_LESS(state.feasibility_test(), epsilon);
        UTEST_CHECK_LESS(state.kkt_optimality_test(), epsilon);
    }
}

static void check_solution_augmented(const function_t& function, const logger_t& logger)
{
    (void)function;
    (void)logger;

    /*// FIXME: It is possible to detect unfeasibility or unboundedness with augmented lagrangian method?!
    if (expected.m_status != solver_status::converged)
    {
        return;
    }

    auto solver                         = solver_augmented_lagrangian_t{};
    solver.parameter("solver::epsilon") = 1e-10;

    const auto function = make_function(program);
    const auto x0       = make_full_tensor<scalar_t>(make_dims(function->size()), 4.0);
    auto       state    = solver.minimize(*function, x0, logger);

    UTEST_CHECK_EQUAL(state.status(), expected.m_status);
    if (expected.m_status == solver_status::converged)
    {
        check_solution(program, expected, state.x(), state.fx(), state.kkt_optimality_test());
    }*/
}

static void check_solution_penalty(const function_t& function, const logger_t& logger)
{
    (void)function;
    (void)logger;

    /*// FIXME: It is possible to detect unfeasibility or unboundedness with the penalty method?!
    if (expected.m_status != solver_status::converged)
    {
        return;
    }

    auto solver                         = solver_quadratic_penalty_t{};
    solver.parameter("solver::epsilon") = 1e-10;

    const auto function = make_function(program);
    const auto x0       = make_full_tensor<scalar_t>(make_dims(function->size()), 4.0);
    auto       state    = solver.minimize(*function, x0, logger);

    UTEST_CHECK_EQUAL(state.status(), expected.m_status);
    if (expected.m_status == solver_status::converged)
    {
        // NB: The penalty method doesn't provide an estimation of the Lagrangian multipliers!
        check_solution(program, expected, state.x(), state.fx(), std::numeric_limits<scalar_t>::quiet_NaN());
    }*/
}

void check_solution(const function_t& function)
{
    // const auto [A, b, G, h] = make_linear_constraints(function);

    // TODO: test duplicated equality constraints
    /*if (program.m_eq.valid())
    {
        auto dprogram = program;
        dprogram.m_eq = duplicate(program.m_eq, 1.0, 0.0);

        check_with_logger([&](const logger_t& logger) { check_solution_penalty(dprogram, expected, logger); });
        check_with_logger([&](const logger_t& logger) { check_solution_program(dprogram, expected, logger); });
        check_with_logger([&](const logger_t& logger) { check_solution_augmented(dprogram, expected, logger); });
    }*/

    // TODO: test linearly dependant equality constraints
    /*if (program.m_eq.valid())
    {
        auto dprogram = program;
        dprogram.m_eq = duplicate(program.m_eq, 0.2, 1.1);

        check_with_logger([&](const logger_t& logger) { check_solution_penalty(dprogram, expected, logger); });
        check_with_logger([&](const logger_t& logger) { check_solution_program(dprogram, expected, logger); });
        check_with_logger([&](const logger_t& logger) { check_solution_augmented(dprogram, expected, logger); });
    }*/

    // test original program
    check_with_logger([&](const logger_t& logger) { check_solution_penalty(function, logger); });
    check_with_logger([&](const logger_t& logger) { check_solution_interior(function, logger); });
    check_with_logger([&](const logger_t& logger) { check_solution_augmented(function, logger); });
}
