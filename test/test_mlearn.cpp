#include "fixture/utils.h"
#include <nano/mlearn/train.h>
#include <nano/mlearn/stacking.h>

using namespace nano;

inline std::ostream& operator<<(std::ostream& stream, train_status status)
{
    return stream << scat(status);
}

UTEST_BEGIN_MODULE()

UTEST_CASE(train_point)
{
    const auto nan = std::numeric_limits<scalar_t>::quiet_NaN();
    {
        const auto point = train_point_t{};
        UTEST_CHECK_EQUAL(point.valid(), false);
    }
    {
        const auto point = train_point_t{1.5, 0.5, 0.6};
        UTEST_CHECK_EQUAL(point.valid(), true);
    }
    {
        const auto point = train_point_t{nan, 0.5, 0.6};
        UTEST_CHECK_EQUAL(point.valid(), false);
    }
    {
        const auto point = train_point_t{1.5, nan, 0.6};
        UTEST_CHECK_EQUAL(point.valid(), false);
    }
    {
        const auto point = train_point_t{1.5, 0.5, nan};
        UTEST_CHECK_EQUAL(point.valid(), false);
    }
    {
        const auto point1 = train_point_t{1.5, 0.5, 0.60};
        const auto point2 = train_point_t{1.4, 0.4, 0.61};
        UTEST_CHECK(point1 < point2);
    }
    {
        const auto point1 = train_point_t{1.5, 0.5, nan};
        const auto point2 = train_point_t{1.4, 0.4, 0.61};
        const auto point3 = train_point_t{1.5, 0.5, nan};
        UTEST_CHECK(point2 < point1);
        UTEST_CHECK(!(point1 < point2));
        UTEST_CHECK(!(point3 < point1));
        UTEST_CHECK(!(point1 < point3));
    }
}

UTEST_CASE(train_curve)
{
    const auto inf = std::numeric_limits<scalar_t>::infinity();
    {
        train_curve_t curve;
        UTEST_CHECK_EQUAL(curve.optindex(), 0U);
        UTEST_CHECK_EQUAL(curve.check(1U), train_status::better);
    }
    {
        train_curve_t curve;
        curve.add(1.5, 0.5, 0.6);
        UTEST_CHECK_EQUAL(curve.optindex(), 0U);
        UTEST_CHECK_EQUAL(curve.check(0U), train_status::better);
        UTEST_CHECK_EQUAL(curve.check(1U), train_status::better);
    }
    {
        train_curve_t curve;
        curve.add(1.5, 0.5, 0.6);
        curve.add(inf, 0.4, 0.5);
        UTEST_CHECK_EQUAL(curve.optindex(), 0U);
        UTEST_CHECK_EQUAL(curve.check(0U), train_status::diverged);
    }
    {
        train_curve_t curve;
        curve.add(1.5, 0.5, 0.6);
        UTEST_CHECK_EQUAL(curve.optindex(), 0U);
        UTEST_CHECK_EQUAL(curve.check(0U), train_status::better);
        UTEST_CHECK_EQUAL(curve.check(1U), train_status::better);
        UTEST_CHECK_EQUAL(curve.check(2U), train_status::better);
        UTEST_CHECK_CLOSE(curve.optimum().vd_error(), 0.6, 1e-12);

        curve.add(1.4, 0.4, 0.5);
        UTEST_CHECK_EQUAL(curve.optindex(), 1U);
        UTEST_CHECK_EQUAL(curve.check(0U), train_status::overfit);
        UTEST_CHECK_EQUAL(curve.check(1U), train_status::better);
        UTEST_CHECK_EQUAL(curve.check(2U), train_status::better);
        UTEST_CHECK_CLOSE(curve.optimum().vd_error(), 0.5, 1e-12);

        curve.add(1.3, 0.3, 0.4);
        UTEST_CHECK_EQUAL(curve.optindex(), 2U);
        UTEST_CHECK_EQUAL(curve.check(0U), train_status::overfit);
        UTEST_CHECK_EQUAL(curve.check(1U), train_status::better);
        UTEST_CHECK_EQUAL(curve.check(2U), train_status::better);
        UTEST_CHECK_CLOSE(curve.optimum().vd_error(), 0.4, 1e-12);

        curve.add(1.2, 0.2, 0.5);
        UTEST_CHECK_EQUAL(curve.optindex(), 2U);
        UTEST_CHECK_EQUAL(curve.check(0U), train_status::overfit);
        UTEST_CHECK_EQUAL(curve.check(1U), train_status::overfit);
        UTEST_CHECK_EQUAL(curve.check(2U), train_status::worse);
        UTEST_CHECK_CLOSE(curve.optimum().vd_error(), 0.4, 1e-12);

        curve.add(1.1, 0.1, 0.6);
        UTEST_CHECK_EQUAL(curve.optindex(), 2U);
        UTEST_CHECK_EQUAL(curve.check(0U), train_status::overfit);
        UTEST_CHECK_EQUAL(curve.check(1U), train_status::overfit);
        UTEST_CHECK_EQUAL(curve.check(2U), train_status::overfit);
        UTEST_CHECK_CLOSE(curve.optimum().vd_error(), 0.4, 1e-12);

        curve.add(1.0, 0.0, 0.7);
        UTEST_CHECK_EQUAL(curve.optindex(), 2U);
        UTEST_CHECK_EQUAL(curve.check(0U), train_status::overfit);
        UTEST_CHECK_EQUAL(curve.check(1U), train_status::overfit);
        UTEST_CHECK_EQUAL(curve.check(2U), train_status::overfit);
        UTEST_CHECK_CLOSE(curve.optimum().vd_error(), 0.4, 1e-12);

        curve.add(inf, 0.0, 0.7);
        UTEST_CHECK_EQUAL(curve.optindex(), 2U);
        UTEST_CHECK_EQUAL(curve.check(7U), train_status::diverged);
        UTEST_CHECK_CLOSE(curve.optimum().vd_error(), 0.4, 1e-12);
    }
    {
        auto curve = train_curve_t{};
        curve.add(2.1, 1.1, 1.4);
        curve.add(2.0, 1.0, 1.3);
        curve.add(1.9, 0.9, 1.2);

        std::stringstream stream1;
        UTEST_CHECK(curve.save(stream1, ',', false));
        UTEST_CHECK_EQUAL(stream1.str(), scat(
            0, ",", 2.1, ",", 1.1, ",", 1.4, "\n",
            1, ",", 2.0, ",", 1.0, ",", 1.3, "\n",
            2, ",", 1.9, ",", 0.9, ",", 1.2, "\n"));

        std::stringstream stream2;
        UTEST_CHECK(curve.save(stream2, ';', true));
        UTEST_CHECK_EQUAL(stream2.str(), scat(
            "step;tr_value;tr_error;vd_error\n",
            0, ";", 2.1, ";", 1.1, ";", 1.4, "\n",
            1, ";", 2.0, ";", 1.0, ";", 1.3, "\n",
            2, ";", 1.9, ";", 0.9, ";", 1.2, "\n"));
    }
}

UTEST_CASE(train_fold)
{
    auto tuning = train_fold_t{};
    UTEST_CHECK(!std::isfinite(tuning.tr_value()));
    UTEST_CHECK(!std::isfinite(tuning.tr_error()));
    UTEST_CHECK(!std::isfinite(tuning.vd_error()));

    auto& curve0 = tuning.add({{"hyper", 0.0}});
    auto& curve1 = tuning.add({{"hyper", 1.0}});
    auto& curve2 = tuning.add({{"hyper", 2.0}});

    curve0.add(2.1, 1.1, 1.4);
    curve0.add(2.0, 1.0, 1.3);
    curve0.add(1.9, 0.9, 1.2);
    curve0.add(1.8, 0.9, 1.3);

    curve1.add(3.1, 2.1, 2.5);
    curve1.add(2.1, 1.1, 2.0);
    curve1.add(1.1, 0.1, 1.5);
    curve1.add(1.1, 0.1, 1.0);

    const auto inf = std::numeric_limits<scalar_t>::infinity();
    const auto nan = std::numeric_limits<scalar_t>::quiet_NaN();
    curve2.add(inf, nan, nan);

    const auto& opt = tuning.optimum();
    UTEST_CHECK_EQUAL(opt.first, scat("hyper=", 1.0, ";"));
    UTEST_CHECK_CLOSE(tuning.tr_value(), 1.1, 1e-12);
    UTEST_CHECK_CLOSE(tuning.tr_error(), 0.1, 1e-12);
    UTEST_CHECK_CLOSE(tuning.vd_error(), 1.0, 1e-12);

    tuning.test(1.1);
    UTEST_CHECK_CLOSE(tuning.te_error(), 1.1, 1e-12);
}

UTEST_CASE(train_result)
{
    auto result = train_result_t{};

    auto& fold0 = result.add();
    auto& hype0 = fold0.add({{"hyper", 0.0}});
    hype0.add(2.1, 1.1, 1.4);
    hype0.add(2.0, 1.0, 1.3);
    hype0.add(1.9, 0.9, 1.2);
    hype0.add(1.8, 0.9, 1.3);
    fold0.test(1.1);

    auto& fold1 = result.add();
    auto& hype1 = fold1.add({{"hyper", 1.0}});
    hype1.add(2.1, 1.1, 1.3);
    hype1.add(2.0, 1.0, 1.1);
    hype1.add(1.9, 0.9, 1.0);
    hype1.add(1.8, 0.7, 0.8);
    fold1.test(1.2);

    auto& fold2 = result.add();
    fold2.test(1.0);

    const auto nan = std::numeric_limits<scalar_t>::quiet_NaN();

    UTEST_CHECK_EQUAL(result.size(), 3U);
    UTEST_CHECK_CLOSE(result[0U].te_error(), 1.1, 1e-12);
    UTEST_CHECK_CLOSE(result[1U].te_error(), 1.2, 1e-12);
    UTEST_CHECK_CLOSE(result[2U].te_error(), 1.0, 1e-12);

    std::stringstream stream1;
    UTEST_CHECK(result.save(stream1, ',', false));
    UTEST_CHECK_EQUAL(stream1.str(), scat(
        0, ",", 0.9, ",", 1.2, ",", 1.1, "\n",
        1, ",", 0.7, ",", 0.8, ",", 1.2, "\n",
        2, ",", nan, ",", nan, ",", 1.0, "\n"));

    std::stringstream stream2;
    UTEST_CHECK(result.save(stream2, ';', true));
    UTEST_CHECK_EQUAL(stream2.str(), scat(
        "fold;tr_error;vd_error;te_error\n",
        0, ";", 0.9, ";", 1.2, ";", 1.1, "\n",
        1, ";", 0.7, ";", 0.8, ";", 1.2, "\n",
        2, ";", nan, ";", nan, ";", 1.0, "\n"));
}

UTEST_CASE(stacking)
{
    const auto loss = make_loss();
    const auto solver = make_solver();

    auto weights = vector_t(3);
    weights(0) = 0.10;
    weights(1) = 0.50;
    weights(2) = 0.40;

    auto targets = tensor4d_t(100, 4, 4, 3);
    auto outputs = tensor5d_t(weights.size(), 100, 4, 4, 3);

    targets.random();
    outputs.random();
    outputs.vector(2) = (targets.vector() - outputs.vector(0) * weights(0) - outputs.vector(1) * weights(1)) / weights(2);

    auto function = stacking_function_t{*loss, targets, outputs};
    UTEST_CHECK_NOTHROW(function.batch(16));
    UTEST_CHECK_EQUAL(function.batch(), 16);
    UTEST_CHECK_EQUAL(function.size(), weights.size());
    UTEST_CHECK_THROW(function.batch(-1), std::runtime_error);

    for (auto trial = 0; trial < 10; ++ trial)
    {
        const vector_t x = vector_t::Random(function.size());
        UTEST_CHECK_LESS(function.grad_accuracy(x), 10 * epsilon2<scalar_t>());
    }

    const auto state = solver->minimize(function, vector_t::Zero(function.size()));
    UTEST_CHECK(state);
    UTEST_CHECK(state.converged(solver->epsilon()));
    UTEST_CHECK_CLOSE(state.f, 0.0, 1e-8);
    UTEST_CHECK_EIGEN_CLOSE(stacking_function_t::as_weights(state.x), weights, 1e+1 * solver->epsilon());
}

UTEST_END_MODULE()
