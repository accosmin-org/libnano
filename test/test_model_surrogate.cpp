#include <utest/utest.h>
#include "fixture/loss.h"
#include "fixture/solver.h"
#include "fixture/function.h"
#include <nano/model/surrogate.h>

using namespace nano;

static void check_minimizer(const function_t& function, const vector_t& optimum)
{
    const auto* const solver_id = function.smooth() ? "lbfgs" : "osga";
    const auto epsilon = 1e-9;

    const auto solver = make_solver(solver_id, epsilon);
    const auto state = check_minimize(*solver, solver_id, function, vector_t::Random(function.size()), 10000, epsilon);
    UTEST_CHECK_CLOSE(state.f, 0.0, 1e-6);
    UTEST_CHECK_CLOSE(state.x, optimum, 1e-7);
}

static void check_surrogate(const tensor1d_t& p, const tensor1d_t& q)
{
    const auto function = quadratic_surrogate_t(q.vector());
    check_gradient(function);
    check_convexity(function);
    check_minimizer(function, p.vector());
    UTEST_CHECK_EQUAL(function.size(), p.size());
}

static void check_surrogate_fit(const tensor1d_t& q, const tensor2d_t& p, const tensor1d_t& y)
{
    for (const auto* const loss_id : {"squared", "absolute"})
    {
        [[maybe_unused]] const auto _ = utest_test_name_t{loss_id};

        const auto loss = make_loss(loss_id);
        const auto function = quadratic_surrogate_fit_t(*loss, p, y);
        check_gradient(function);
        check_convexity(function);
        check_minimizer(function, q.vector());
        UTEST_CHECK_EQUAL(function.size(), q.size());
    }
}

UTEST_BEGIN_MODULE(test_model_surrogate)

UTEST_CASE(quadratic_surrogate_1d)
{
    const auto p = make_tensor<scalar_t>(make_dims(1), +1.0);
    const auto q = make_tensor<scalar_t>(make_dims(3), +1.0, -2.0, +1.0);

    check_surrogate(p, q);
}

UTEST_CASE(quadratic_surrogate_2d)
{
    const auto p = make_tensor<scalar_t>(make_dims(2), +1.0, -2.0);
    const auto q = make_tensor<scalar_t>(make_dims(6), +5.0, -2.0, +4.0, +1.0, +0.0, +1.0);

    check_surrogate(p, q);
}

UTEST_CASE(quadratic_surrogate_2dc)
{
    const auto p = make_tensor<scalar_t>(make_dims(2), 0.1, 1.0);
    const auto q = make_tensor<scalar_t>(make_dims(6), 1.0, 0.0, -2.0, 1.0, -0.2, 1.01);

    check_surrogate(p, q);
}

UTEST_CASE(quadratic_surrogate_fit1d)
{
    const auto q = make_tensor<scalar_t>(make_dims(3), 1.0, 0.5, -1.0);
    const auto p = make_tensor<scalar_t>(make_dims(7, 1),
        -3.0, -2.0, -1.0, +0.0, +1.0, +2.0, +3.0
    );

    tensor1d_t y(7);
    for (tensor_size_t i = 0; i < y.size(); ++ i)
    {
        const auto p0 = p(i, 0);
        y(i) = q(0) * 1.0 + q(1) * p0 + q(2) * p0 * p0;
    }

    check_surrogate_fit(q, p, y);
}

UTEST_CASE(quadratic_surrogate_fit2d)
{
    const auto q = make_tensor<scalar_t>(make_dims(6), 1.0, 0.5, 1.5, 2.0, -1.0, -1.0);
    const auto p = make_tensor<scalar_t>(make_dims(25, 2),
        -2.0, -2.0, -2.0, -1.0, -2.0, +0.0, -2.0, +1.0, -2.0, +2.0,
        -1.0, -2.0, -1.0, -1.0, -1.0, +0.0, -1.0, +1.0, -1.0, +2.0,
        +0.0, -2.0, +0.0, -1.0, +0.0, +0.0, +0.0, +1.0, +0.0, +2.0,
        +1.0, -2.0, +1.0, -1.0, +1.0, +0.0, +1.0, +1.0, +1.0, +2.0,
        +2.0, -2.0, +2.0, -1.0, +2.0, +0.0, +2.0, +1.0, +2.0, +2.0
    );

    tensor1d_t y(25);
    for (tensor_size_t i = 0; i < y.size(); ++ i)
    {
        const auto p0 = p(i, 0), p1 = p(i, 1);
        y(i) = q(0) * 1.0 + q(1) * p0 + q(2) * p1 + q(3) * p0 * p0 + q(4) * p0 * p1 + q(5) * p1 * p1;
    }

    check_surrogate_fit(q, p, y);
}

UTEST_END_MODULE()
