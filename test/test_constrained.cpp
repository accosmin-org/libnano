#include "fixture/function.h"
#include <nano/function/constraints.h>
#include <nano/function/penalty.h>
#include <nano/solver/penalty.h>

using namespace nano;

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

template <typename tpenalty>
static void check_penalty(const function_t& constrained, bool expected_convexity, bool expected_smoothness)
{
    auto penalty = tpenalty{constrained};

    for (const auto penalty_term : {1e-1, 1e+0, 1e+1, 1e+2, 1e+3})
    {
        penalty.penalty_term(penalty_term);

        check_gradient(penalty);

        UTEST_CHECK_EQUAL(penalty.strong_convexity(), 0.0);
        UTEST_CHECK_EQUAL(penalty.convex(), expected_convexity);
        UTEST_CHECK_EQUAL(penalty.smooth(), expected_smoothness);
    }
}

static void check_penalties(const function_t& constrained, bool expected_convexity, bool expected_smoothness)
{
    check_penalty<linear_penalty_function_t>(constrained, expected_convexity,
                                             constrained.constraints().empty() ? expected_smoothness : false);
    check_penalty<quadratic_penalty_function_t>(constrained, expected_convexity, expected_smoothness);
}

template <typename tpenalty>
static void check_penalty(const function_t& constrained, const vector_t& x, bool expected_valid)
{
    UTEST_CHECK_EQUAL(constrained.valid(x), expected_valid);

    auto penalty = tpenalty{constrained};

    for (const auto penalty_term : {1e-1, 1e+0, 1e+1, 1e+2, 1e+3})
    {
        const auto fx = constrained.vgrad(x);
        const auto qx = penalty.penalty_term(penalty_term).vgrad(x);
        if (expected_valid)
        {
            UTEST_CHECK_CLOSE(fx, qx, 1e-15);
        }
        else
        {
            UTEST_CHECK_LESS(fx + 1e-6, qx);
        }
    }
}

static void check_penalties(const function_t& constrained, const vector_t& x, bool expected_valid)
{
    check_penalty<linear_penalty_function_t>(constrained, x, expected_valid);
    check_penalty<quadratic_penalty_function_t>(constrained, x, expected_valid);
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

UTEST_BEGIN_MODULE(test_constrained)

UTEST_CASE(constraint_minimum)
{
    const auto constraint = constraint_t{
        minimum_t{1.0, 0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(make_x(0.0, 1.0), constraint), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(0.9, 1.0), constraint), 0.1, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(1.0, 0.0), constraint), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(2.0, 0.0), constraint), 0.0, 1e-15);
}

UTEST_CASE(constraint_maximum)
{
    const auto constraint = constraint_t{
        maximum_t{1.0, 1}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(make_x(0.0, 0.0), constraint), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(0.9, 0.9), constraint), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(1.0, 1.0), constraint), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(2.0, 1.2), constraint), 0.2, 1e-15);
}

UTEST_CASE(constraint_equality)
{
    const auto constraint = constraint_t{equality_t{std::make_unique<sumabsm1_function_t>(3)}};

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(!::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(make_x(0.0, 0.0, 0.0), constraint), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(0.9, 0.9, 0.0), constraint), 0.8, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(1.0, 1.0, 0.0), constraint), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(2.0, 1.2, 0.0), constraint), 2.2, 1e-15);
}

UTEST_CASE(constraint_inequality)
{
    const auto constraint = constraint_t{inequality_t{std::make_unique<sumabsm1_function_t>(3)}};

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(!::nano::smooth(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(make_x(0.0, 0.0, 0.0), constraint), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(0.5, 0.2, 0.0), constraint), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(0.5, 0.0, -0.5), constraint), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(1.0, 1.2, 0.0), constraint), 1.2, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(make_x(0.1, -0.7, -0.4), constraint), 0.2, 1e-15);
}

UTEST_CASE(noconstraint_sum)
{
    auto constrained = sum_function_t{3};
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 0U);

    check_penalties(constrained, true, true);
    for (auto trial = 0; trial < 100; ++trial)
    {
        check_penalties(constrained, vector_t::Random(3), true);
    }
}

UTEST_CASE(noconstraint_sumabsm1)
{
    auto constrained = sumabsm1_function_t{3};
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 0U);

    check_penalties(constrained, true, false);
    for (auto trial = 0; trial < 100; ++trial)
    {
        check_penalties(constrained, vector_t::Random(3), true);
    }
}

UTEST_CASE(constrained_box1)
{
    auto constrained = sum_function_t{3};
    UTEST_CHECK(!constrained.constrain(make_box_constraints(-0.5, -0.5, 3)));
    UTEST_CHECK(!constrained.constrain(make_box_constraints(+0.5, -0.5, 3)));
    UTEST_CHECK(!constrained.constrain(make_box_constraints(-0.5, +0.5, 0)));
    UTEST_CHECK(!constrained.constrain(make_box_constraints(-0.5, +0.5, 4)));
    UTEST_CHECK(constrained.constrain(make_box_constraints(-0.5, +0.5, 3)));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 6U);

    check_penalties(constrained, true, true);
    check_penalties(constrained, make_x(-0.1, -0.1, -0.1), true);
    check_penalties(constrained, make_x(+0.2, +0.2, +0.2), true);
    check_penalties(constrained, make_x(+0.5, +0.5, +0.5), true);
    check_penalties(constrained, make_x(-0.7, -0.7, -0.7), false);
    check_penalties(constrained, make_x(+0.8, +0.8, +0.8), false);
    check_penalties(constrained, make_x(-0.2, +0.1, +0.0), true);
    check_penalties(constrained, make_x(-0.2, +0.2, -0.7), false);
    check_penalties(constrained, make_x(-0.2, +0.6, +0.0), false);
}

UTEST_CASE(constrained_box2)
{
    auto constrained = sum_function_t{3};
    UTEST_CHECK(!constrained.constrain(make_box_constraints(make_x(-0.5, -0.5, -0.5, -0.5), make_x(+0.5, +0.5, +0.5))));
    UTEST_CHECK(!constrained.constrain(make_box_constraints(make_x(-0.5, -0.5, -0.5), make_x(+0.5, +0.5, +0.5, +0.5))));
    UTEST_CHECK(!constrained.constrain(make_box_constraints(make_x(+0.5, +0.5, +0.5), make_x(-0.5, -0.5, -0.5))));
    UTEST_CHECK(constrained.constrain(make_box_constraints(make_x(-0.5, -0.5, -0.5), make_x(+0.5, +0.5, +0.5))));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 6U);

    check_penalties(constrained, true, true);
    check_penalties(constrained, make_x(-0.2, +0.1, +0.0), true);
    check_penalties(constrained, make_x(-0.2, +0.1, +0.0), true);
    check_penalties(constrained, make_x(-0.2, +0.6, +0.0), false);
    check_penalties(constrained, make_x(-0.2, -0.9, +1.0), false);
}

UTEST_CASE(constrained_ball_inequality)
{
    auto constrained = sum_function_t{3};
    UTEST_CHECK(!constrained.constrain(make_ball_inequality_constraint(make_x(1.0, 1.0, 1.0, 1.0), 1.0)));
    UTEST_CHECK(!constrained.constrain(make_ball_inequality_constraint(make_x(1.0, 1.0, 1.0), 0.0)));
    UTEST_CHECK(constrained.constrain(make_ball_inequality_constraint(make_x(0.0, 0.0, 0.0), 1.0)));
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
    UTEST_CHECK(!constrained.constrain(make_affine_equality_constraint(make_x(1.0, 1.0, 1.0, 1.0), -3.0)));
    UTEST_CHECK(constrained.constrain(make_affine_equality_constraint(make_x(1.0, 1.0, 1.0), -3.0)));
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
    UTEST_CHECK(!constrained.constrain(make_affine_inequality_constraint(make_x(1.0, 1.0, 1.0, 1.0), -3.0)));
    UTEST_CHECK(constrained.constrain(make_affine_inequality_constraint(make_x(1.0, 1.0, 1.0), -3.0)));
    UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

    check_penalties(constrained, true, false);
    check_penalties(constrained, make_x(0.1, 0.2, 0.3), true);
    check_penalties(constrained, make_x(0.1, 1.2, 1.3), true);
    check_penalties(constrained, make_x(0.5, 1.5, 2.5), false);
}

UTEST_CASE(constrained_cauchy_inequality)
{
    auto constrained = cauchy_function_t{3};
    UTEST_CHECK(!constrained.constrain(make_inequality_constraint(std::make_unique<cauchy_function_t>(4))));
    UTEST_CHECK(constrained.constrain(make_inequality_constraint(std::make_unique<cauchy_function_t>(3))));
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
    UTEST_CHECK(!constrained.constrain(make_equality_constraint(std::make_unique<sumabsm1_function_t>(4))));
    UTEST_CHECK(constrained.constrain(make_equality_constraint(std::make_unique<sumabsm1_function_t>(3))));
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
    UTEST_CHECK(!constrained.constrain(make_inequality_constraint(std::make_unique<sumabsm1_function_t>(4))));
    UTEST_CHECK(constrained.constrain(make_inequality_constraint(std::make_unique<sumabsm1_function_t>(3))));
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
        UTEST_CHECK(!constrained.constrain(make_quadratic_inequality_constraint(P2x2, q3, 1.0)));
        UTEST_CHECK(!constrained.constrain(make_quadratic_inequality_constraint(P2x3, q2, 1.0)));
        UTEST_CHECK(!constrained.constrain(make_quadratic_inequality_constraint(P3x2, q2, 1.0)));
        UTEST_CHECK(constrained.constrain(make_quadratic_inequality_constraint(P2x2, q2, 1.0)));
        UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

        check_penalties(constrained, false, true);
    }
    {
        auto constrained = sum_function_t{3};
        UTEST_CHECK(!constrained.constrain(make_quadratic_inequality_constraint(P3x3, q4, 1.0)));
        UTEST_CHECK(!constrained.constrain(make_quadratic_inequality_constraint(P3x4, q3, 1.0)));
        UTEST_CHECK(!constrained.constrain(make_quadratic_inequality_constraint(P4x3, q3, 1.0)));
        UTEST_CHECK(constrained.constrain(make_quadratic_inequality_constraint(P3x3, q3, 1.0)));
        UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

        check_penalties(constrained, true, true);
    }
    {
        auto constrained = sum_function_t{3};
        UTEST_CHECK(!constrained.constrain(make_quadratic_equality_constraint(P3x3, q4, 1.0)));
        UTEST_CHECK(!constrained.constrain(make_quadratic_equality_constraint(P3x4, q3, 1.0)));
        UTEST_CHECK(!constrained.constrain(make_quadratic_equality_constraint(P4x3, q3, 1.0)));
        UTEST_CHECK(constrained.constrain(make_quadratic_equality_constraint(P3x3, q3, 1.0)));
        UTEST_CHECK_EQUAL(constrained.constraints().size(), 1U);

        check_penalties(constrained, false, true);
    }
}

UTEST_CASE(minimize_case0)
{
    auto function = sum_function_t{2};
    function.constrain(make_ball_equality_constraint(make_x(0.0, 0.0), std::sqrt(2.0)));

    const auto base_solver = solver_t::all().get("lbfgs");

    {
        const auto solver = solver_linear_penalty_t{};
        solver.minimize(*base_solver, function, make_x(0.0, 0.0));
    }
    {
        const auto solver = solver_quadratic_penalty_t{};
        solver.minimize(*base_solver, function, make_x(0.0, 0.0));
    }
}

UTEST_END_MODULE()
