#include <nano/function/program.h>
#include <nano/program/benchmark.h>
#include <nano/program/solver.h>
#include <nano/solver/augmented.h>
#include <nano/solver/penalty.h>
#include <utest/utest.h>

using namespace nano;

inline auto make_permutation(const tensor_size_t m)
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
}

template <class tprogram>
void check_solution_program(tprogram program, const program::expected_t& expected, const logger_t& logger)
{
    auto solver = program::solver_t{};

    auto state =
        (expected.m_x0.size() > 0) ? solver.solve(program, expected.m_x0, logger) : solver.solve(program, logger);

    UTEST_CHECK_EQUAL(state.m_status, expected.m_status);
    if (expected.m_status == solver_status::converged)
    {
        check_solution(program, expected, state.m_x, state.m_fx, state.m_kkt);
    }
}

template <class tprogram>
void check_solution_augmented(const tprogram& program, const program::expected_t& expected, const logger_t& logger)
{
    // FIXME: It is possible to detect unfeasibility or unboundedness with augmented lagrangian method?!
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
    }
}

template <class tprogram>
void check_solution_penalty(const tprogram& program, const program::expected_t& expected, const logger_t& logger)
{
    // FIXME: It is possible to detect unfeasibility or unboundedness with the penalty method?!
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
    }
}

template <class tprogram>
void check_solution(const tprogram& program, const program::expected_t& expected)
{
    // TODO: move this to std::remove_cvref_t when moving C++20!
    if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<tprogram>>, program::quadratic_program_t>)
    {
        UTEST_REQUIRE(program.convex());
    }

    // test duplicated equality constraints
    if (program.m_eq.valid())
    {
        auto dprogram = program;
        dprogram.m_eq = duplicate(program.m_eq, 1.0, 0.0);

        check_with_logger([&](const logger_t& logger) { check_solution_penalty(dprogram, expected, logger); });
        check_with_logger([&](const logger_t& logger) { check_solution_program(dprogram, expected, logger); });
        check_with_logger([&](const logger_t& logger) { check_solution_augmented(dprogram, expected, logger); });
    }

    // test linearly dependant equality constraints
    if (program.m_eq.valid())
    {
        auto dprogram = program;
        dprogram.m_eq = duplicate(program.m_eq, 0.2, 1.1);

        check_with_logger([&](const logger_t& logger) { check_solution_penalty(dprogram, expected, logger); });
        check_with_logger([&](const logger_t& logger) { check_solution_program(dprogram, expected, logger); });
        check_with_logger([&](const logger_t& logger) { check_solution_augmented(dprogram, expected, logger); });
    }

    // test original program
    check_with_logger([&](const logger_t& logger) { check_solution_penalty(program, expected, logger); });
    check_with_logger([&](const logger_t& logger) { check_solution_program(program, expected, logger); });
    check_with_logger([&](const logger_t& logger) { check_solution_augmented(program, expected, logger); });
}
