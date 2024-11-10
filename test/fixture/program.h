#include <nano/function/program.h>
#include <nano/program/benchmark.h>
#include <nano/program/solver.h>
#include <nano/solver/augmented.h>
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
void check_solution_program(tprogram program, const program::expected_t& expected)
{
    const auto failures = utest_n_failures.load();

    program.reduce();

    auto solver = program::solver_t{};
    auto stream = std::ostringstream{};
    auto logger = make_stream_logger(stream);

    auto bstate =
        (expected.m_x0.size() > 0) ? solver.solve(program, expected.m_x0, logger) : solver.solve(program, logger);

    UTEST_CHECK_EQUAL(bstate.m_status, expected.m_status);
    if (expected.m_status == solver_status::converged)
    {
        UTEST_CHECK(program.feasible(bstate.m_x, expected.m_epsilon));
        if (expected.m_xbest.size() > 0) // NB: sometimes the solution is not known analytically!
        {
            UTEST_CHECK_CLOSE(bstate.m_x, expected.m_xbest, expected.m_epsilon);
        }
        if (expected.m_vbest.size() > 0)
        {
            UTEST_CHECK_CLOSE(bstate.m_v, expected.m_vbest, expected.m_epsilon);
        }
        if (std::isfinite(expected.m_fbest))
        {
            UTEST_CHECK_CLOSE(bstate.m_fx, expected.m_fbest, expected.m_epsilon);
        }
    }

    if (failures != utest_n_failures.load())
    {
        std::cout << stream.str();
    }
}

template <class tprogram>
void check_solution_augmented(const tprogram& program, const program::expected_t& expected)
{
    // FIXME: It is possible to detect unfeasibility or unboundedness with augmented lagrangian method?!
    if (expected.m_status != solver_status::converged)
    {
        return;
    }

    const auto failures = utest_n_failures.load();

    auto solver                         = solver_augmented_lagrangian_t{};
    solver.parameter("solver::epsilon") = 1e-10;

    auto stream = std::ostringstream{};
    auto logger = make_stream_logger(stream);

    const auto function = make_function(program);
    const auto x0       = make_full_tensor<scalar_t>(make_dims(function->size()), 4.0);
    auto       bstate   = solver.minimize(*function, x0, logger);

    UTEST_CHECK_EQUAL(bstate.status(), expected.m_status);
    if (expected.m_status == solver_status::converged)
    {
        UTEST_CHECK(program.feasible(bstate.x(), expected.m_epsilon));
        if (expected.m_xbest.size() > 0) // NB: sometimes the solution is not known analytically!
        {
            UTEST_CHECK_CLOSE(bstate.x(), expected.m_xbest, expected.m_epsilon);
        }
        if (std::isfinite(expected.m_fbest))
        {
            UTEST_CHECK_CLOSE(bstate.fx(), expected.m_fbest, expected.m_epsilon);
        }
        // TODO: check the lagrange multipliers as well for the augmented lagrangian solver!
    }

    if (failures != utest_n_failures.load())
    {
        std::cout << stream.str();
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

        // NB: the Lagrange multipliers are not kept when duplicating equality constraints!
        auto dexpected    = expected;
        dexpected.m_vbest = vector_t{};

        check_solution_program(dprogram, dexpected);
        check_solution_augmented(dprogram, dexpected);
    }

    // test linearly dependant equality constraints
    if (program.m_eq.valid())
    {
        auto dprogram = program;
        dprogram.m_eq = duplicate(program.m_eq, 0.2, 1.1);

        // NB: the Lagrange multipliers are not kept when adding linearly dependant equality constraints!
        auto dexpected    = expected;
        dexpected.m_vbest = vector_t{};

        check_solution_program(dprogram, dexpected);
        check_solution_augmented(dprogram, dexpected);
    }

    // test original program
    check_solution_augmented(program, expected);
    check_solution_program(program, expected);
}
