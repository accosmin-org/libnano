#include "fixture/function.h"
#include <nano/function/penalty.h>
#include <nano/solver/penalty.h>

using namespace nano;
using namespace nano::constraint;

template <typename... tvalues>
static vector_t make_x(tvalues... values)
{
    return make_tensor<scalar_t, 1>(make_dims(sizeof...(values)), values...).vector();
}

template <tensor_size_t trows, typename... tvalues>
static matrix_t make_X(tvalues... values)
{
    return make_tensor<scalar_t, 1>(make_dims(sizeof...(values)), values...).reshape(trows, -1).matrix();
}

static auto make_solver(const char* solver_id, scalar_t epsilon = 1e-9, int max_evals = 5000)
{
    auto solver = solver_t::all().get(solver_id);
    UTEST_REQUIRE(solver != nullptr);

    solver->lsearchk("cgdescent");
    solver->parameter("solver::epsilon")   = epsilon;
    solver->parameter("solver::max_evals") = max_evals;
    return solver;
}

template <typename tpenalty>
static void check_penalty(const function_t& function, bool expected_convexity, bool expected_smoothness)
{
    auto penalty_function = tpenalty{function};

    for (const auto penalty : {1e-1, 1e+0, 1e+1, 1e+2, 1e+3})
    {
        penalty_function.penalty(penalty);

        check_gradient(penalty_function);
        UTEST_CHECK_EQUAL(penalty_function.strong_convexity(), 0.0);
        UTEST_CHECK_EQUAL(penalty_function.convex(), expected_convexity);
        UTEST_CHECK_EQUAL(penalty_function.smooth(), expected_smoothness);
    }
}

static void check_penalties(const function_t& function, bool expected_convexity, bool expected_smoothness)
{
    const auto unconstrained = function.constraints().empty();

    check_penalty<linear_penalty_function_t>(function, expected_convexity, unconstrained ? expected_smoothness : false);
    check_penalty<quadratic_penalty_function_t>(function, expected_convexity, expected_smoothness);
    check_penalty<linear_quadratic_penalty_function_t>(function, expected_convexity, expected_smoothness);
}

template <typename tpenalty>
static void check_penalty(const function_t& function, const vector_t& x, bool expected_valid)
{
    UTEST_CHECK_EQUAL(function.valid(x), expected_valid);

    auto penalty_function = tpenalty{function};

    for (const auto penalty : {1e-1, 1e+0, 1e+1, 1e+2, 1e+3})
    {
        const auto fx = function.vgrad(x);
        const auto op = [&]()
        {
            const auto qx = penalty_function.penalty(penalty).vgrad(x);
            if (expected_valid)
            {
                UTEST_CHECK_CLOSE(fx, qx, 1e-16);
            }
            else
            {
                UTEST_CHECK_LESS(fx + 1e-8, qx);
            }
        };

        if constexpr (std::is_same_v<tpenalty, linear_quadratic_penalty_function_t>)
        {
            for (const auto smoothing : {1e-2, 1e+0, 1e+2})
            {
                penalty_function.smoothing(smoothing);
                op();
            }
        }
        else
        {
            op();
        }
    }
}

static void check_penalties(const function_t& function, const vector_t& x, bool expected_valid)
{
    check_penalty<linear_penalty_function_t>(function, x, expected_valid);
    check_penalty<quadratic_penalty_function_t>(function, x, expected_valid);
    check_penalty<linear_quadratic_penalty_function_t>(function, x, expected_valid);
}

static void check_minimize(solver_penalty_t& penalty_solver, solver_t& solver, const function_t& function,
                           const vector_t& xbest, const scalar_t fbest, const scalar_t epsilon)
{
    const auto x0 = make_random_x0(function, 5.0);

    std::stringstream stream;
    stream << std::fixed << std::setprecision(12) << "x0=" << x0.transpose() << "." << std::endl;

    solver.logger(
        [&](const auto& state)
        {
            stream << std::fixed << std::setprecision(12) << state << ",x=" << state.x.transpose() << "." << std::endl;
            return true;
        });

    penalty_solver.logger(
        [&](const auto& state)
        {
            stream << std::fixed << std::setprecision(12) << state << ",x=" << state.x.transpose() << "." << std::endl;
            return true;
        });

    const auto state = penalty_solver.minimize(solver, function, x0);

    const auto old_n_failures = utest_n_failures.load();

    UTEST_CHECK_CLOSE(state.f, fbest, epsilon);
    UTEST_CHECK_CLOSE(state.x, xbest, epsilon);
    UTEST_CHECK_EQUAL(state.status, solver_status::converged);
    UTEST_CHECK_LESS(state.p.sum(), penalty_solver.parameter("solver::penalty::epsilon").value<scalar_t>());

    if (old_n_failures != utest_n_failures.load())
    {
        std::cout << stream.str() << std::endl;
    }
}

static void check_penalty_solver(const function_t& function, const vector_t& xbest, const scalar_t fbest)
{
    auto lsolver  = solver_linear_penalty_t{};
    auto qsolver  = solver_quadratic_penalty_t{};
    auto lqsolver = solver_linear_quadratic_penalty_t{};

    for (const auto* const solver_id : {"ellipsoid"})
    {
        if (!linear_penalty_function_t{function}.convex())
        {
            // NB: cannot solver non-convex non-smooth problems precisely!
            continue;
        }

        UTEST_NAMED_CASE(scat(function.name(), "_linear_penalty_solver_", solver_id));

        const auto ref_solver = make_solver(solver_id);
        for (tensor_size_t trial = 0; trial < 10; ++trial)
        {
            check_minimize(lsolver, *ref_solver, function, xbest, fbest, 1e-4);
        }
    }

    for (const auto* const solver_id : {"lbfgs"})
    {
        UTEST_NAMED_CASE(scat(function.name(), "_quadratic_penalty_solver_", solver_id));

        const auto ref_solver = make_solver(solver_id);
        for (tensor_size_t trial = 0; trial < 10; ++trial)
        {
            check_minimize(qsolver, *ref_solver, function, xbest, fbest, 1e-6);
        }
    }

    for (const auto* const solver_id : {"lbfgs"})
    {
        UTEST_NAMED_CASE(scat(function.name(), "_linear_quadratic_penalty_solver_", solver_id));

        const auto ref_solver = make_solver(solver_id);
        for (tensor_size_t trial = 0; trial < 10; ++trial)
        {
            check_minimize(lqsolver, *ref_solver, function, xbest, fbest, 1e-6);
        }
    }
}

class sum_function_t final : public function_t
{
public:
    explicit sum_function_t(tensor_size_t size)
        : function_t("sum", size)
    {
        convex(true);
        smooth(true);
    }

    scalar_t do_vgrad(const vector_t& x, vector_t* gx) const override
    {
        if (gx != nullptr)
        {
            gx->noalias() = vector_t::Ones(x.size());
        }
        return x.sum();
    }
};

class cauchy_function_t final : public function_t
{
public:
    explicit cauchy_function_t(tensor_size_t size)
        : function_t("cauchy", size)
    {
        convex(false);
        smooth(true);
    }

    scalar_t do_vgrad(const vector_t& x, vector_t* gx) const override
    {
        if (gx != nullptr)
        {
            gx->noalias() = 2.0 * x / (0.36 + x.dot(x));
        }
        return std::log(0.36 + x.dot(x));
    }
};

class sumabsm1_function_t final : public function_t
{
public:
    explicit sumabsm1_function_t(tensor_size_t size)
        : function_t("sumabsm1", size)
    {
        convex(true);
        smooth(false);
    }

    scalar_t do_vgrad(const vector_t& x, vector_t* gx) const override
    {
        if (gx != nullptr)
        {
            gx->array() = x.array().sign();
        }
        return x.array().abs().sum() - 1.0;
    }
};

class objective1_function_t final : public function_t
{
public:
    explicit objective1_function_t()
        : function_t("objective1", 2)
    {
        convex(true);
        smooth(true);
    }

    scalar_t do_vgrad(const vector_t& x, vector_t* gx) const override
    {
        if (gx != nullptr)
        {
            gx->noalias() = vector_t::Ones(x.size());
        }
        return x.sum();
    }
};

class objective2_function_t final : public function_t
{
public:
    explicit objective2_function_t()
        : function_t("objective2", 2)
    {
        convex(true);
        smooth(true);
    }

    scalar_t do_vgrad(const vector_t& x, vector_t* gx) const override
    {
        if (gx != nullptr)
        {
            (*gx)(0) = -10.0 * x(0);
            (*gx)(1) = +2.0 * x(1);
        }
        return -5.0 * x(0) * x(0) + x(1) * x(1);
    }
};

class objective3_function_t final : public function_t
{
public:
    explicit objective3_function_t()
        : function_t("objective3", 1)
    {
        convex(true);
        smooth(true);
    }

    scalar_t do_vgrad(const vector_t& x, vector_t* gx) const override
    {
        if (gx != nullptr)
        {
            gx->noalias() = vector_t::Ones(x.size());
        }
        return x.sum();
    }
};

class objective4_function_t final : public function_t
{
public:
    explicit objective4_function_t()
        : function_t("objective4", 2)
    {
        convex(true);
        smooth(true);
    }

    scalar_t do_vgrad(const vector_t& x, vector_t* gx) const override
    {
        if (gx != nullptr)
        {
            (*gx)(0) = 4.0 * x(0) - 1.0;
            (*gx)(1) = 4.0 * x(1);
        }
        return 2.0 * (x(0) * x(0) + x(1) * x(1) - 1.0) - x(0);
    }
};

UTEST_BEGIN_MODULE(test_constrained)

UTEST_CASE(minimum)
{
    const auto constraint = constraint_t{
        minimum_t{1.0, 0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 1.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.9, 1.0)), 0.1, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(2.0, 0.0)), 0.0, 1e-15);
}

UTEST_CASE(maximum)
{
    const auto constraint = constraint_t{
        maximum_t{1.0, 1}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.9, 0.9)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(2.0, 1.2)), 0.2, 1e-15);
}

UTEST_CASE(constant)
{
    const auto constraint = constraint_t{
        constant_t{1.0, 1}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.9, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 1.1)), 0.1, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(2.0, 0.8)), 0.2, 1e-15);
}

UTEST_CASE(euclidean_ball_equality)
{
    const auto constraint = constraint_t{
        euclidean_ball_equality_t{make_x(0.0, 0.0), 1.0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 2.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 2.0)), 4.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 0.0)), 1.0, 1e-15);
}

UTEST_CASE(euclidean_ball_inequality)
{
    const auto constraint = constraint_t{
        euclidean_ball_inequality_t{make_x(0.0, 0.0), 1.0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 2.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 2.0)), 4.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 0.0)), 0.0, 1e-15);
}

UTEST_CASE(linear_equality)
{
    const auto constraint = constraint_t{
        linear_equality_t{make_x(1.0, 1.0), -2.0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 0.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 2.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(2.0, 2.0)), 2.0, 1e-15);
}

UTEST_CASE(linear_inequality)
{
    const auto constraint = constraint_t{
        linear_inequality_t{make_x(1.0, 1.0), -2.0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 2.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(2.0, 2.0)), 2.0, 1e-15);
}

UTEST_CASE(quadratic_equality)
{
    const auto constraint = constraint_t{
        quadratic_equality_t{make_X<2>(1.0, 2.0, 2.0, 1.0), make_x(1.0, 1.0), -5.0}
    };

    UTEST_CHECK(!::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 0.0)), 5.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 1.0)), 3.5, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 0.0)), 3.5, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 2.0)), 4.5, 1e-15);
}

UTEST_CASE(quadratic_inequality)
{
    const auto constraint = constraint_t{
        quadratic_inequality_t{make_X<3>(2.0, -1., 0.0, -1., 2.0, -1., 0.0, -1., 2.0), make_x(1.0, 1.0, 1.0), -2.0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 2.0 - std::sqrt(2.0), 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 0.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 0.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 1.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 1.0, 1.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 0.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 0.0, 1.0)), 2.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 1.0, 0.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 1.0, 1.0)), 2.0, 1e-15);
}

UTEST_CASE(functional_equality)
{
    const auto constraint = constraint_t{functional_equality_t{std::make_unique<sumabsm1_function_t>(3)}};

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(!::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 0.0, 0.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.9, 0.9, 0.0)), 0.8, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 1.0, 0.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(2.0, 1.2, 0.0)), 2.2, 1e-15);
}

UTEST_CASE(functional_inequality)
{
    const auto constraint = constraint_t{functional_inequality_t{std::make_unique<sumabsm1_function_t>(3)}};

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(!::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.0, 0.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.5, 0.2, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.5, 0.0, -0.5)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(1.0, 1.2, 0.0)), 1.2, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_x(0.1, -0.7, -0.4)), 0.2, 1e-15);
}

UTEST_CASE(noconstraint_convex_smooth)
{
    auto constrained = sum_function_t{3};
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 0U);

    check_penalties(constrained, true, true);
    for (auto trial = 0; trial < 100; ++trial)
    {
        check_penalties(constrained, make_random_x0(constrained), true);
    }
}

UTEST_CASE(noconstraint_convex_nonsmooth)
{
    auto constrained = sumabsm1_function_t{3};
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 0U);

    check_penalties(constrained, true, false);
    for (auto trial = 0; trial < 100; ++trial)
    {
        check_penalties(constrained, make_random_x0(constrained), true);
    }
}

UTEST_CASE(noconstraint_nonconvex_smooth)
{
    auto constrained = cauchy_function_t{3};
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 0U);

    check_penalties(constrained, false, true);
    for (auto trial = 0; trial < 100; ++trial)
    {
        check_penalties(constrained, make_random_x0(constrained), true);
    }
}

UTEST_CASE(constrained_box_one)
{
    auto constrained = sum_function_t{3};
    UTEST_CHECK(!constrained.constrain(-0.5, +0.5, -1));
    UTEST_CHECK(!constrained.constrain(-0.5, +0.5, +3));
    UTEST_CHECK(!constrained.constrain(+0.5, -0.5, +3));
    UTEST_CHECK(constrained.constrain(-0.5, +0.5, +2));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 2U);

    check_penalties(constrained, true, true);
    check_penalties(constrained, make_x(-0.1, -0.1, -0.1), true);
    check_penalties(constrained, make_x(+0.2, +0.2, +0.2), true);
    check_penalties(constrained, make_x(+0.5, +0.5, +0.5), true);
    check_penalties(constrained, make_x(-0.7, -0.7, -0.7), false);
    check_penalties(constrained, make_x(+0.8, +0.8, +0.8), false);
    check_penalties(constrained, make_x(-0.7, +0.1, +0.0), true);
    check_penalties(constrained, make_x(-0.2, +0.2, -0.7), false);
    check_penalties(constrained, make_x(-0.2, +0.6, +0.0), true);
}

UTEST_CASE(constrained_box_all)
{
    auto constrained = sum_function_t{3};
    UTEST_CHECK(!constrained.constrain(+0.5, -0.5));
    UTEST_CHECK(constrained.constrain(-0.5, +0.5));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 6U);

    check_penalties(constrained, true, true);
    check_penalties(constrained, make_x(-0.2, +0.1, +0.0), true);
    check_penalties(constrained, make_x(-0.2, +0.1, +0.4), true);
    check_penalties(constrained, make_x(-0.2, +0.6, +0.0), false);
    check_penalties(constrained, make_x(-0.2, -0.3, +1.0), false);
}

UTEST_CASE(constrained_box_vector)
{
    auto constrained = sum_function_t{3};
    UTEST_CHECK(!constrained.constrain(make_x(-0.5, -0.5, -0.5, -0.5), make_x(+0.5, +0.5, +0.5)));
    UTEST_CHECK(!constrained.constrain(make_x(-0.5, -0.5, -0.5), make_x(+0.5, +0.5, +0.5, +0.5)));
    UTEST_CHECK(!constrained.constrain(make_x(+0.5, +0.5, +0.5), make_x(-0.5, -0.5, -0.5)));
    UTEST_CHECK(constrained.constrain(make_x(-0.5, -0.5, -0.5), make_x(+0.5, +0.5, +0.5)));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 6U);

    check_penalties(constrained, true, true);
    check_penalties(constrained, make_x(-0.2, +0.1, +0.0), true);
    check_penalties(constrained, make_x(-0.2, +0.1, +0.4), true);
    check_penalties(constrained, make_x(-0.2, +0.6, +0.0), false);
    check_penalties(constrained, make_x(-0.2, -0.3, +1.0), false);
}

UTEST_CASE(constrained_constant)
{
    auto constrained = sumabsm1_function_t{3};
    UTEST_CHECK(!constrained.constrain(constant_t{1.0, -1}));
    UTEST_CHECK(!constrained.constrain(constant_t{1.0, +3}));
    UTEST_CHECK(constrained.constrain(constant_t{1.0, 2}));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

    check_penalties(constrained, true, false);
    check_penalties(constrained, make_x(0.5, 1.5, 1.0), true);
    check_penalties(constrained, make_x(1.0, 1.0, 1.0), true);
    check_penalties(constrained, make_x(0.1, 0.2, 0.3), false);
    check_penalties(constrained, make_x(0.1, 1.2, 1.3), false);
    check_penalties(constrained, make_x(0.5, 1.5, 2.5), false);
}

UTEST_CASE(constrained_euclidean_ball_inequality)
{
    auto constrained = sum_function_t{3};
    UTEST_CHECK(!constrained.constrain(euclidean_ball_inequality_t{make_x(1.0, 1.0, 1.0, 1.0), 1.0}));
    UTEST_CHECK(!constrained.constrain(euclidean_ball_inequality_t{make_x(1.0, 1.0, 1.0), 0.0}));
    UTEST_CHECK(constrained.constrain(euclidean_ball_inequality_t{make_x(0.0, 0.0, 0.0), 1.0}));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

    check_penalties(constrained, true, true);
    check_penalties(constrained, make_x(0.0, 0.0, 0.0), true);
    check_penalties(constrained, make_x(0.5, 0.5, 0.5), true);
    check_penalties(constrained, make_x(0.6, 0.6, 0.6), false);
    check_penalties(constrained, make_x(1.0, 1.0, 1.0), false);
}

UTEST_CASE(constrained_affine_equality)
{
    auto constrained = sumabsm1_function_t{3};
    UTEST_CHECK(!constrained.constrain(linear_equality_t{make_x(1.0, 1.0, 1.0, 1.0), -3.0}));
    UTEST_CHECK(constrained.constrain(linear_equality_t{make_x(1.0, 1.0, 1.0), -3.0}));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

    check_penalties(constrained, true, false);
    check_penalties(constrained, make_x(0.5, 1.5, 1.0), true);
    check_penalties(constrained, make_x(1.0, 1.0, 1.0), true);
    check_penalties(constrained, make_x(0.1, 0.2, 0.3), false);
    check_penalties(constrained, make_x(0.1, 1.2, 1.3), false);
    check_penalties(constrained, make_x(0.5, 1.5, 2.5), false);
}

UTEST_CASE(constrained_affine_inequality)
{
    auto constrained = sumabsm1_function_t{3};
    UTEST_CHECK(!constrained.constrain(linear_inequality_t{make_x(1.0, 1.0, 1.0, 1.0), -3.0}));
    UTEST_CHECK(constrained.constrain(linear_inequality_t{make_x(1.0, 1.0, 1.0), -3.0}));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

    check_penalties(constrained, true, false);
    check_penalties(constrained, make_x(0.1, 0.2, 0.3), true);
    check_penalties(constrained, make_x(0.1, 1.2, 1.3), true);
    check_penalties(constrained, make_x(0.5, 1.5, 2.5), false);
}

UTEST_CASE(constrained_cauchy_inequality)
{
    auto constrained = cauchy_function_t{3};
    UTEST_CHECK(!constrained.constrain(functional_inequality_t{std::make_unique<cauchy_function_t>(4)}));
    UTEST_CHECK(constrained.constrain(functional_inequality_t{std::make_unique<cauchy_function_t>(3)}));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

    check_penalties(constrained, false, true);
    check_penalties(constrained, make_x(0.0, 0.0, 0.0), true);
    check_penalties(constrained, make_x(0.0, 0.0, 0.7), true);
    check_penalties(constrained, make_x(0.8, 0.0, 0.0), true);
    check_penalties(constrained, make_x(0.1, 0.2, 0.3), true);
    check_penalties(constrained, make_x(0.8, 0.1, 0.0), false);
    check_penalties(constrained, make_x(0.0, 0.9, 0.0), false);
}

UTEST_CASE(constrained_sumabsm1_equality)
{
    auto constrained = sum_function_t{3};
    UTEST_CHECK(!constrained.constrain(functional_equality_t{std::make_unique<sumabsm1_function_t>(4)}));
    UTEST_CHECK(constrained.constrain(functional_equality_t{std::make_unique<sumabsm1_function_t>(3)}));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

    check_penalties(constrained, false, false);
    check_penalties(constrained, make_x(0.0, 0.0, 1.0), true);
    check_penalties(constrained, make_x(-0.9, 0.1, 0.0), true);
    check_penalties(constrained, make_x(0.0, 0.9, 0.0), false);
    check_penalties(constrained, make_x(-0.6, +0.8, 0.1), false);
    check_penalties(constrained, make_x(-1.6, +0.8, 0.1), false);
}

UTEST_CASE(constrained_sumabsm1_inequality)
{
    auto constrained = sum_function_t{3};
    UTEST_CHECK(!constrained.constrain(functional_inequality_t{std::make_unique<sumabsm1_function_t>(4)}));
    UTEST_CHECK(constrained.constrain(functional_inequality_t{std::make_unique<sumabsm1_function_t>(3)}));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

    check_penalties(constrained, true, false);
    check_penalties(constrained, make_x(0.0, 0.0, 1.0), true);
    check_penalties(constrained, make_x(0.0, 0.9, 0.0), true);
    check_penalties(constrained, make_x(-0.6, +0.2, 0.1), true);
    check_penalties(constrained, make_x(-1.6, +0.8, 0.1), false);
    check_penalties(constrained, make_x(-0.2, +0.8, 0.1), false);
}

UTEST_CASE(constrained_quadratic)
{
    auto q2 = make_x(1.0, 1.0);
    auto q3 = make_x(1.0, 1.0, 1.0);
    auto q4 = make_x(1.0, 1.0, 1.0, 1.0);

    auto P2x2 = make_X<2>(1.0, 2.0, 2.0, 1.0);
    auto P2x3 = make_X<2>(1.0, 2.0, 2.0, 1.0, 1.0, 1.0);
    auto P3x2 = make_X<3>(1.0, 2.0, 2.0, 1.0, 1.0, 1.0);

    auto P3x3 = make_X<3>(2.0, -1., 0.0, -1., 2.0, -1., 0.0, -1., 2.0);
    auto P3x4 = make_X<3>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    auto P4x3 = make_X<4>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

    {
        auto constrained = sum_function_t{2};
        UTEST_CHECK(!constrained.constrain(quadratic_inequality_t{P2x2, q3, 1.0}));
        UTEST_CHECK(!constrained.constrain(quadratic_inequality_t{P2x3, q2, 1.0}));
        UTEST_CHECK(!constrained.constrain(quadratic_inequality_t{P3x2, q2, 1.0}));
        UTEST_CHECK(constrained.constrain(quadratic_inequality_t{P2x2, q2, 1.0}));
        UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

        check_penalties(constrained, false, true);
    }
    {
        auto constrained = sum_function_t{3};
        UTEST_CHECK(!constrained.constrain(quadratic_inequality_t{P3x3, q4, 1.0}));
        UTEST_CHECK(!constrained.constrain(quadratic_inequality_t{P3x4, q3, 1.0}));
        UTEST_CHECK(!constrained.constrain(quadratic_inequality_t{P4x3, q3, 1.0}));
        UTEST_CHECK(constrained.constrain(quadratic_inequality_t{P3x3, q3, 1.0}));
        UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

        check_penalties(constrained, true, true);
    }
    {
        auto constrained = sum_function_t{3};
        UTEST_CHECK(!constrained.constrain(quadratic_equality_t{P3x3, q4, 1.0}));
        UTEST_CHECK(!constrained.constrain(quadratic_equality_t{P3x4, q3, 1.0}));
        UTEST_CHECK(!constrained.constrain(quadratic_equality_t{P4x3, q3, 1.0}));
        UTEST_CHECK(constrained.constrain(quadratic_equality_t{P3x3, q3, 1.0}));
        UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

        check_penalties(constrained, false, true);
    }
}

UTEST_CASE(minimize_objective1)
{
    // see 17.3, "Numerical optimization", Nocedal & Wright, 2nd edition
    auto function = objective1_function_t{};
    function.constrain(euclidean_ball_equality_t{make_x(0.0, 0.0), std::sqrt(2.0)});

    const auto fbest = -2.0;
    const auto xbest = make_x(-1.0, -1.0);

    check_penalty_solver(function, xbest, fbest);
}

UTEST_CASE(minimize_objective2)
{
    // see 17.5, "Numerical optimization", Nocedal & Wright, 2nd edition
    auto function = objective2_function_t{};
    function.constrain(constant_t{1.0, 0});

    const auto fbest = -5.0;
    const auto xbest = make_x(1.0, 0.0);

    check_penalty_solver(function, xbest, fbest);
}

UTEST_CASE(minimize_objective3)
{
    // see 17.24, "Numerical optimization", Nocedal & Wright, 2nd edition
    auto function = objective3_function_t{};
    function.constrain(minimum_t{1.0, 0});

    const auto fbest = +1.0;
    const auto xbest = make_x(1.0);

    check_penalty_solver(function, xbest, fbest);
}

UTEST_CASE(minimize_objective4)
{
    // see 15.34, "Numerical optimization", Nocedal & Wright, 2nd edition
    auto function = objective4_function_t{};
    function.constrain(euclidean_ball_equality_t{make_x(0.0, 0.0), 1.0});

    const auto fbest = -1.0;
    const auto xbest = make_x(1.0, 0.0);

    check_penalty_solver(function, xbest, fbest);
}

// TODO: check the case when the constraints are not feasible - is it possible to detect this case?!
// TODO: check that it works with various gamma {2.0, 5.0, 10.0}

UTEST_END_MODULE()
