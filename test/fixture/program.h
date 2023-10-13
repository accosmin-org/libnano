#include <Eigen/Dense>
#include <nano/program/solver.h>
#include <utest/utest.h>

using namespace nano;
using namespace nano::program;

static auto make_logger(std::ostringstream& stream, const int stop_at_iters = -1)
{
    return [&stream, stop_at_iters = stop_at_iters](const solver_state_t& state)
    {
        stream << std::fixed << std::setprecision(12) << "i=" << state.m_iters << ",fx=" << state.m_fx
               << ",eta=" << state.m_eta << ",rdual=" << state.m_rdual.lpNorm<Eigen::Infinity>()
               << ",rcent=" << state.m_rcent.lpNorm<Eigen::Infinity>()
               << ",rprim=" << state.m_rprim.lpNorm<Eigen::Infinity>() << ",rcond=" << state.m_ldlt_rcond
               << (state.m_ldlt_positive ? "(+)" : "(-)") << "[" << state.m_status << "]" << std::endl;
        return state.m_iters != stop_at_iters;
    };
}

static auto make_permutation(const tensor_size_t m)
{
    auto permutation = arange(0, m);
    std::shuffle(permutation.begin(), permutation.end(), make_rng());
    return permutation;
}

static auto duplicate(const equality_t& equality, const scalar_t dep_w1, const scalar_t dep_w2)
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

    return equality_t{A2, b2};
}

struct expected_t
{
    expected_t() = default;

    explicit expected_t(vector_t xbest)
        : m_xbest(std::move(xbest))
    {
    }

    auto& x0(vector_t x0)
    {
        m_x0 = std::move(x0);
        return *this;
    }

    auto& status(solver_status status)
    {
        m_status = status;
        return *this;
    }

    auto& epsilon(const scalar_t epsilon)
    {
        m_epsilon = epsilon;
        return *this;
    }

    auto& use_logger(const bool use_logger)
    {
        m_use_logger = use_logger;
        return *this;
    }

    auto& fbest(const scalar_t fbest)
    {
        m_fbest = fbest;
        return *this;
    }

    auto make_solver(std::ostringstream& stream) const
    {
        const auto stop_at_iters = m_status == solver_status::stopped ? 3 : -1;

        if (m_use_logger)
        {
            return solver_t{make_logger(stream, stop_at_iters)};
        }
        else
        {
            return solver_t{};
        }
    }

    vector_t      m_xbest;
    scalar_t      m_fbest{std::numeric_limits<scalar_t>::quiet_NaN()};
    vector_t      m_x0;
    scalar_t      m_epsilon{1e-8};
    solver_status m_status{solver_status::converged};
    bool          m_use_logger{true};
};

template <typename tprogram>
auto check_solution_(tprogram program, const expected_t& expected)
{
    const auto failures = utest_n_failures.load();

    program.reduce();

    auto stream = std::ostringstream{};
    auto solver = expected.make_solver(stream);
    auto bstate = (expected.m_x0.size() > 0) ? solver.solve(program, expected.m_x0) : solver.solve(program);

    UTEST_CHECK_EQUAL(bstate.m_status, expected.m_status);
    if (expected.m_status == solver_status::converged)
    {
        UTEST_CHECK(program.feasible(bstate.m_x, expected.m_epsilon));
        if (expected.m_xbest.size() > 0) // NB: sometimes the solution is not known analytically!
        {
            UTEST_CHECK_CLOSE(bstate.m_x, expected.m_xbest, expected.m_epsilon);
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

    return bstate;
}

template <typename tprogram>
auto check_solution(const tprogram& program, const expected_t& expected)
{
    // test duplicated equality constraints
    if (program.m_eq)
    {
        auto dprogram = program;
        dprogram.m_eq = duplicate(program.m_eq, 1.0, 0.0);
        check_solution_(dprogram, expected);
    }

    // test linearly dependant equality constraints
    if (program.m_eq)
    {
        auto dprogram = program;
        dprogram.m_eq = duplicate(program.m_eq, 0.2, 1.1);
        check_solution_(dprogram, expected);
    }

    // test original program
    return check_solution_(program, expected);
}
