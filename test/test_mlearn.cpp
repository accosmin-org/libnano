#include <utest/utest.h>
#include <nano/mlearn/split.h>
#include <nano/mlearn/train.h>
#include <nano/mlearn/cluster.h>
#include <nano/mlearn/feature.h>
#include <nano/mlearn/elemwise.h>

using namespace nano;

inline std::ostream& operator<<(std::ostream& stream, importance type)
{
    return stream << scat(type);
}

inline std::ostream& operator<<(std::ostream& stream, train_status status)
{
    return stream << scat(status);
}

template <typename tscalar>
std::ostream& operator<<(std::ostream& stream, const std::vector<tscalar>& values)
{
    stream << "{";
    for (const auto& value : values)
    {
        stream << "{" << value << "}";
    }
    return stream << "}";
}

UTEST_BEGIN_MODULE(test_mlearn)

UTEST_CASE(split2)
{
    const auto count = 120;
    const auto percentage1 = 60;
    const auto percentage2 = 100 - percentage1;

    indices_t set1, set2;
    std::tie(set1, set2) = nano::split2(count, percentage1);

    UTEST_CHECK_EQUAL(set1.size(), percentage1 * count / 100);
    UTEST_CHECK_EQUAL(set2.size(), percentage2 * count / 100);

    UTEST_CHECK_GREATER_EQUAL(set1.min(), 0);
    UTEST_CHECK_GREATER_EQUAL(set2.min(), 0);

    UTEST_CHECK_LESS(set1.max(), count);
    UTEST_CHECK_LESS(set2.max(), count);

    UTEST_CHECK(std::is_sorted(begin(set1), end(set1)));
    UTEST_CHECK(std::is_sorted(begin(set2), end(set2)));

    UTEST_CHECK(std::adjacent_find(begin(set1), end(set1)) == end(set1));
    UTEST_CHECK(std::adjacent_find(begin(set2), end(set2)) == end(set2));
}

UTEST_CASE(split3)
{
    const auto count = 120;
    const auto percentage1 = 60;
    const auto percentage2 = 30;
    const auto percentage3 = 100 - percentage1 - percentage2;

    indices_t set1, set2, set3;
    std::tie(set1, set2, set3) = nano::split3(count, percentage1, percentage2);

    UTEST_CHECK_EQUAL(set1.size(), percentage1 * count / 100);
    UTEST_CHECK_EQUAL(set2.size(), percentage2 * count / 100);
    UTEST_CHECK_EQUAL(set3.size(), percentage3 * count / 100);

    UTEST_CHECK_GREATER_EQUAL(set1.min(), 0);
    UTEST_CHECK_GREATER_EQUAL(set2.min(), 0);
    UTEST_CHECK_GREATER_EQUAL(set3.min(), 0);

    UTEST_CHECK_LESS(set1.max(), count);
    UTEST_CHECK_LESS(set2.max(), count);
    UTEST_CHECK_LESS(set3.max(), count);

    UTEST_CHECK(std::is_sorted(begin(set1), end(set1)));
    UTEST_CHECK(std::is_sorted(begin(set2), end(set2)));
    UTEST_CHECK(std::is_sorted(begin(set3), end(set3)));

    UTEST_CHECK(std::adjacent_find(begin(set1), end(set1)) == end(set1));
    UTEST_CHECK(std::adjacent_find(begin(set2), end(set2)) == end(set2));
    UTEST_CHECK(std::adjacent_find(begin(set3), end(set3)) == end(set3));
}

UTEST_CASE(sample_with_replacement)
{
    for (auto trial = 0; trial < 100; ++ trial)
    {
        const auto indices = nano::sample_with_replacement(120, 50);

        UTEST_CHECK_EQUAL(indices.size(), 60);
        UTEST_CHECK_LESS(indices.max(), 120);
        UTEST_CHECK_GREATER_EQUAL(indices.min(), 0);
        UTEST_CHECK(std::is_sorted(begin(indices), end(indices)));
    }
}

UTEST_CASE(sample_without_replacement)
{
    for (auto trial = 0; trial < 100; ++ trial)
    {
        const auto indices = nano::sample_without_replacement(140, 50);

        UTEST_CHECK_EQUAL(indices.size(), 70);
        UTEST_CHECK_LESS(indices.max(), 140);
        UTEST_CHECK_GREATER_EQUAL(indices.min(), 0);
        UTEST_CHECK(std::is_sorted(begin(indices), end(indices)));
        UTEST_CHECK(std::adjacent_find(begin(indices), end(indices)) == end(indices));
    }
}

UTEST_CASE(sample_without_replacement_all)
{
    const auto indices = nano::sample_without_replacement(100, 100);

    UTEST_CHECK_EQUAL(indices, arange(0, 100));
}

UTEST_CASE(split_valid2)
{
    const auto split = split_t{nano::split2(80, 60), arange(80, 100)};

    UTEST_CHECK(split.valid(100));
    UTEST_CHECK_EQUAL(split.indices(protocol::train).size(), 48);
    UTEST_CHECK_EQUAL(split.indices(protocol::valid).size(), 32);
    UTEST_CHECK_EQUAL(split.indices(protocol::test).size(), 20);

    UTEST_CHECK(!split.valid(90));
}

UTEST_CASE(split_valid3)
{
    const auto split = split_t{nano::split3(100, 60, 30)};

    UTEST_CHECK(split.valid(100));
    UTEST_CHECK_EQUAL(split.indices(protocol::train).size(), 60);
    UTEST_CHECK_EQUAL(split.indices(protocol::valid).size(), 30);
    UTEST_CHECK_EQUAL(split.indices(protocol::test).size(), 10);

    UTEST_CHECK(!split.valid(90));
}

UTEST_CASE(split_invalid_empty)
{
    const auto split = split_t{};

    UTEST_CHECK(!split.valid(100));
}

UTEST_CASE(split_invalid_tr_out_of_range)
{
    auto split = split_t{nano::split3(100, 60, 30)};

    split.indices(protocol::train)(0) = -1;
    UTEST_CHECK(!split.valid(100));

    split.indices(protocol::train)(0) = 101;
    UTEST_CHECK(!split.valid(100));
}

UTEST_CASE(split_invalid_vd_out_of_range)
{
    auto split = split_t{nano::split3(100, 60, 30)};

    split.indices(protocol::valid)(0) = -1;
    UTEST_CHECK(!split.valid(100));

    split.indices(protocol::valid)(0) = 101;
    UTEST_CHECK(!split.valid(100));
}

UTEST_CASE(split_invalid_te_out_of_range)
{
    auto split = split_t{nano::split3(100, 60, 30)};

    split.indices(protocol::test)(0) = -1;
    UTEST_CHECK(!split.valid(100));

    split.indices(protocol::test)(0) = 101;
    UTEST_CHECK(!split.valid(100));
}

UTEST_CASE(split_invalid_tr_vd_intersects)
{
    auto split = split_t{nano::split3(100, 60, 30)};
    split.indices(protocol::valid) = split.indices(protocol::train).slice(0, split.indices(protocol::valid).size());

    UTEST_CHECK(!split.valid(100));
}

UTEST_CASE(split_invalid_tr_te_intersects)
{
    auto split = split_t{nano::split3(100, 60, 30)};
    split.indices(protocol::test) = split.indices(protocol::train).slice(0, split.indices(protocol::test).size());

    UTEST_CHECK(!split.valid(100));
}

UTEST_CASE(scale)
{
    const std::array<scalar_t, 3> min = {{-1, +1, +2}};
    const std::array<scalar_t, 3> max = {{+1, +3, +7}};
    const std::array<scalar_t, 3> mean = {{0.0, 1.1, 5.1}};
    const std::array<scalar_t, 3> stdev = {{0.1, 0.2, 0.8}};

    elemwise_stats_t stats;
    stats.set(
        map_tensor(min.data(), 3, 1, 1),
        map_tensor(max.data(), 3, 1, 1),
        map_tensor(mean.data(), 3, 1, 1),
        map_tensor(stdev.data(), 3, 1, 1));

    const std::array<scalar_t, 15> original =
    {{
        -1.0, +1.0, +2.0,
        -0.5, +1.5, +3.0,
        -0.0, +2.0, +4.0,
        +0.5, +2.5, +5.0,
        +1.0, +3.0, +6.0
    }};

    const std::array<scalar_t, 15> normed_none =
    {{
        -1.0, +1.0, +2.0,
        -0.5, +1.5, +3.0,
        -0.0, +2.0, +4.0,
        +0.5, +2.5, +5.0,
        +1.0, +3.0, +6.0
    }};

    const std::array<scalar_t, 15> normed_mean =
    {{
        -0.50, -0.05, -3.1/5.0,
        -0.25, +0.20, -2.1/5.0,
        -0.00, +0.45, -1.1/5.0,
        +0.25, +0.70, -0.1/5.0,
        +0.50, +0.95, +0.9/5.0
    }};

    const std::array<scalar_t, 15> normed_minmax =
    {{
        +0.00, +0.00, +0.00,
        +0.25, +0.25, +0.20,
        +0.50, +0.50, +0.40,
        +0.75, +0.75, +0.60,
        +1.00, +1.00, +0.80
    }};

    const std::array<scalar_t, 15> normed_standard =
    {{
        -10.0, -0.50, -3.1/0.8,
         -5.0, +2.00, -2.1/0.8,
         +0.0, +4.50, -1.1/0.8,
         +5.0, +7.00, -0.1/0.8,
        +10.0, +9.50, +0.9/0.8
    }};

    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_NOTHROW(stats.scale(normalization::none, inputs));
        UTEST_CHECK_EIGEN_CLOSE(inputs.vector(), map_vector(normed_none.data(), 15), epsilon1<scalar_t>());
    }
    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_NOTHROW(stats.scale(normalization::mean, inputs));
        UTEST_CHECK_EIGEN_CLOSE(inputs.vector(), map_vector(normed_mean.data(), 15), epsilon1<scalar_t>());
    }
    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_NOTHROW(stats.scale(normalization::minmax, inputs));
        UTEST_CHECK_EIGEN_CLOSE(inputs.vector(), map_vector(normed_minmax.data(), 15), epsilon1<scalar_t>());
    }
    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_NOTHROW(stats.scale(normalization::standard, inputs));
        UTEST_CHECK_EIGEN_CLOSE(inputs.vector(), map_vector(normed_standard.data(), 15), epsilon1<scalar_t>());
    }
    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_THROW(stats.scale(static_cast<normalization>(-1), inputs), std::runtime_error);
    }
}

UTEST_CASE(cluster)
{
    {
        const auto split = cluster_t{};
        UTEST_CHECK_EQUAL(split.groups(), 0);
        UTEST_CHECK_EQUAL(split.samples(), 0);
    }
    {
        const auto split = cluster_t{7};
        UTEST_CHECK_EQUAL(split.groups(), 1);
        UTEST_CHECK_EQUAL(split.count(0), 0);
        UTEST_CHECK_EQUAL(split.samples(), 7);
    }
    {
        auto split = cluster_t{7, 3};
        UTEST_CHECK_EQUAL(split.groups(), 3);
        UTEST_CHECK_EQUAL(split.count(0), 0);
        UTEST_CHECK_EQUAL(split.count(1), 0);
        UTEST_CHECK_EQUAL(split.count(2), 0);
        UTEST_CHECK_EQUAL(split.samples(), 7);

        split.assign(0, 0);
        split.assign(1, 0);
        split.assign(2, 1);
        split.assign(3, 1);
        split.assign(4, 2);
        split.assign(5, 2);
        split.assign(6, 1);

        UTEST_CHECK_EQUAL(split.groups(), 3);
        UTEST_CHECK_EQUAL(split.count(0), 2);
        UTEST_CHECK_EQUAL(split.count(1), 3);
        UTEST_CHECK_EQUAL(split.count(2), 2);
        UTEST_CHECK_EQUAL(split.samples(), 7);
        UTEST_CHECK_EQUAL(split.group(0), 0);
        UTEST_CHECK_EQUAL(split.group(1), 0);
        UTEST_CHECK_EQUAL(split.group(2), 1);
        UTEST_CHECK_EQUAL(split.group(3), 1);
        UTEST_CHECK_EQUAL(split.group(4), 2);
        UTEST_CHECK_EQUAL(split.group(5), 2);
        UTEST_CHECK_EQUAL(split.group(6), 1);

        split.assign(4, 1);
        split.assign(5, 1);
        split.assign(6, 2);
        split.assign(6, 1);

        UTEST_CHECK_EQUAL(split.groups(), 3);
        UTEST_CHECK_EQUAL(split.count(0), 2);
        UTEST_CHECK_EQUAL(split.count(1), 5);
        UTEST_CHECK_EQUAL(split.count(2), 0);
        UTEST_CHECK_EQUAL(split.samples(), 7);

        const auto indices0 = split.indices(0);
        const auto indices1 = split.indices(1);
        const auto indices2 = split.indices(2);

        UTEST_REQUIRE_EQUAL(indices0.size(), 2);
        UTEST_CHECK_EQUAL(indices0(0), 0);
        UTEST_CHECK_EQUAL(indices0(1), 1);

        UTEST_REQUIRE_EQUAL(indices1.size(), 5);
        UTEST_CHECK_EQUAL(indices1(0), 2);
        UTEST_CHECK_EQUAL(indices1(1), 3);
        UTEST_CHECK_EQUAL(indices1(2), 4);
        UTEST_CHECK_EQUAL(indices1(3), 5);
        UTEST_CHECK_EQUAL(indices1(4), 6);

        UTEST_REQUIRE_EQUAL(indices2.size(), 0);
    }
    {
        auto indices = indices_t{3};
        indices(0) = 0;
        indices(1) = 4;
        indices(2) = 5;

        auto split = cluster_t{7, indices};
        UTEST_CHECK_EQUAL(split.groups(), 1);
        UTEST_CHECK_EQUAL(split.count(0), 3);
        UTEST_CHECK_EQUAL(split.samples(), 7);

        indices_t all_indices(7);
        all_indices.constant(-1);
        split.loop(0, [&] (const tensor_size_t index) { all_indices(index) = +1; });
        UTEST_CHECK_EQUAL(all_indices(0), +1);
        UTEST_CHECK_EQUAL(all_indices(1), -1);
        UTEST_CHECK_EQUAL(all_indices(2), -1);
        UTEST_CHECK_EQUAL(all_indices(3), -1);
        UTEST_CHECK_EQUAL(all_indices(4), +1);
        UTEST_CHECK_EQUAL(all_indices(5), +1);
        UTEST_CHECK_EQUAL(all_indices(6), -1);

        split.assign(3, 0);

        UTEST_CHECK_EQUAL(split.groups(), 1);
        UTEST_CHECK_EQUAL(split.count(0), 4);
        UTEST_CHECK_EQUAL(split.samples(), 7);

        split.loop(0, [&] (const tensor_size_t index) { all_indices(index) = +1; });
        UTEST_CHECK_EQUAL(all_indices(0), +1);
        UTEST_CHECK_EQUAL(all_indices(1), -1);
        UTEST_CHECK_EQUAL(all_indices(2), -1);
        UTEST_CHECK_EQUAL(all_indices(3), +1);
        UTEST_CHECK_EQUAL(all_indices(4), +1);
        UTEST_CHECK_EQUAL(all_indices(5), +1);
        UTEST_CHECK_EQUAL(all_indices(6), -1);
    }
}

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

    auto& curve0 = tuning.add("hyper0");
    auto& curve1 = tuning.add("hyper1");
    auto& curve2 = tuning.add("hyper2");

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
    UTEST_CHECK_EQUAL(opt.first, "hyper1");
    UTEST_CHECK_CLOSE(tuning.tr_value(), 1.1, 1e-12);
    UTEST_CHECK_CLOSE(tuning.tr_error(), 0.1, 1e-12);
    UTEST_CHECK_CLOSE(tuning.vd_error(), 1.0, 1e-12);

    tuning.test(1.1);
    UTEST_CHECK_CLOSE(tuning.te_error(), 1.1, 1e-12);

    tuning.avg_test(1.7);
    UTEST_CHECK_CLOSE(tuning.avg_te_error(), 1.7, 1e-12);
}

UTEST_CASE(train_result)
{
    auto result = train_result_t{};

    auto& fold0 = result.add();
    auto& hype0 = fold0.add("hyper0");
    hype0.add(2.1, 1.1, 1.4);
    hype0.add(2.0, 1.0, 1.3);
    hype0.add(1.9, 0.9, 1.2);
    hype0.add(1.8, 0.9, 1.3);
    fold0.test(1.1);
    fold0.avg_test(1.1);

    auto& fold1 = result.add();
    auto& hype1 = fold1.add("hyper1");
    hype1.add(2.1, 1.1, 1.3);
    hype1.add(2.0, 1.0, 1.1);
    hype1.add(1.9, 0.9, 1.0);
    hype1.add(1.8, 0.7, 0.8);
    fold1.test(1.2);
    fold1.avg_test(1.0);

    auto& fold2 = result.add();
    fold2.test(1.0);
    fold2.avg_test(0.9);

    const auto nan = std::numeric_limits<scalar_t>::quiet_NaN();

    UTEST_CHECK_EQUAL(result.size(), 3U);
    UTEST_CHECK_CLOSE(result[0U].te_error(), 1.1, 1e-12);
    UTEST_CHECK_CLOSE(result[1U].te_error(), 1.2, 1e-12);
    UTEST_CHECK_CLOSE(result[2U].te_error(), 1.0, 1e-12);

    std::stringstream stream1;
    UTEST_CHECK(result.save(stream1, ',', false));
    UTEST_CHECK_EQUAL(stream1.str(), scat(
        0, ",", 0.9, ",", 1.2, ",", 1.1, ",", 1.1, "\n",
        1, ",", 0.7, ",", 0.8, ",", 1.2, ",", 1.0, "\n",
        2, ",", nan, ",", nan, ",", 1.0, ",", 0.9, "\n"));

    std::stringstream stream2;
    UTEST_CHECK(result.save(stream2, ';', true));
    UTEST_CHECK_EQUAL(stream2.str(), scat(
        "fold;tr_error;vd_error;te_error;avg_te_error\n",
        0, ";", 0.9, ";", 1.2, ";", 1.1, ";", 1.1, "\n",
        1, ";", 0.7, ";", 0.8, ";", 1.2, ";", 1.0, "\n",
        2, ";", nan, ";", nan, ";", 1.0, ";", 0.9, "\n"));
}

UTEST_CASE(feature_default)
{
    feature_t feature;
    UTEST_CHECK_EQUAL(static_cast<bool>(feature), false);

    feature = feature_t{"feature"};
    UTEST_CHECK_EQUAL(static_cast<bool>(feature), true);

    UTEST_CHECK(feature_t::missing(feature_t::placeholder_value()));
    UTEST_CHECK(!feature_t::missing(0));
}

UTEST_CASE(feature_compare)
{
    const auto make_feature_cont = [] (const string_t& name)
    {
        auto feature = feature_t{name};
        UTEST_CHECK(!feature.discrete());
        UTEST_CHECK(!feature.optional());
        UTEST_CHECK_THROW(feature.label(0), std::invalid_argument);
        UTEST_CHECK_THROW(feature.label(feature_t::placeholder_value()), std::invalid_argument);
        return feature;
    };

    const auto make_feature_cont_opt = [] (const string_t& name)
    {
        auto feature = feature_t{name}.placeholder("?");
        UTEST_CHECK(!feature.discrete());
        UTEST_CHECK(feature.optional());
        UTEST_CHECK_THROW(feature.label(0), std::invalid_argument);
        UTEST_CHECK_THROW(feature.label(feature_t::placeholder_value()), std::invalid_argument);
        return feature;
    };

    const auto make_feature_cate = [] (const string_t& name)
    {
        auto feature = feature_t{name}.labels({"cate0", "cate1", "cate2"});
        UTEST_CHECK(feature.discrete());
        UTEST_CHECK(!feature.optional());
        UTEST_CHECK_EQUAL(feature.label(0), "cate0");
        UTEST_CHECK_EQUAL(feature.label(1), "cate1");
        UTEST_CHECK_EQUAL(feature.label(2), "cate2");
        UTEST_CHECK_THROW(feature.label(-1), std::out_of_range);
        UTEST_CHECK_THROW(feature.label(+3), std::out_of_range);
        UTEST_CHECK_EQUAL(feature.label(feature_t::placeholder_value()), string_t());
        return feature;
    };

    const auto make_feature_cate_opt = [] (const string_t& name)
    {
        auto feature = feature_t{name}.labels({"cate_opt0", "cate_opt1"}).placeholder("?");
        UTEST_CHECK(feature.discrete());
        UTEST_CHECK(feature.optional());
        UTEST_CHECK_EQUAL(feature.label(0), "cate_opt0");
        UTEST_CHECK_EQUAL(feature.label(1), "cate_opt1");
        UTEST_CHECK_THROW(feature.label(-1), std::out_of_range);
        UTEST_CHECK_THROW(feature.label(+2), std::out_of_range);
        UTEST_CHECK_EQUAL(feature.label(feature_t::placeholder_value()), string_t());
        return feature;
    };

    const auto to_string = [] (const feature_t& feature)
    {
        std::stringstream stream;
        stream << feature;
        return stream.str();
    };

    UTEST_CHECK_EQUAL(make_feature_cont("f"), make_feature_cont("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cont("gf"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cont("f")), "name=f,labels[],placeholder=");

    UTEST_CHECK_EQUAL(make_feature_cont_opt("f"), make_feature_cont_opt("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont_opt("f"), make_feature_cont_opt("ff"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cont_opt("f")), "name=f,labels[],placeholder=?");

    UTEST_CHECK_EQUAL(make_feature_cate("f"), make_feature_cate("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cate("f"), make_feature_cate("x"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cate("f")), "name=f,labels[cate0,cate1,cate2],placeholder=");

    UTEST_CHECK_EQUAL(make_feature_cate_opt("f"), make_feature_cate_opt("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cate_opt("f"), make_feature_cate_opt("x"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cate_opt("f")), "name=f,labels[cate_opt0,cate_opt1],placeholder=?");

    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cate("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cont_opt("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cate_opt("f"));
}

UTEST_END_MODULE()
