#include <nano/core/reduce.h>
#include <nano/wlearner/accumulator.h>
#include <nano/wlearner/criterion.h>
#include <nano/wlearner/util.h>
#include <numbers>
#include <utest/utest.h>

using namespace nano;

struct cache_t
{
    cache_t() = default;

    cache_t(scalar_t score, tensor_size_t index)
        : m_score(score)
        , m_index(index)
    {
    }

    cache_t& operator+=(const cache_t& other)
    {
        m_score += other.m_score;
        return *this;
    }

    cache_t& operator/=(tensor_size_t samples)
    {
        m_score /= static_cast<scalar_t>(samples);
        return *this;
    }

    scalar_t      m_score{0};
    tensor_size_t m_index{0};
};

namespace
{
auto make_accumulator()
{
    const auto tdims = make_dims(1, 1, 1);

    auto acc0 = wlearner::accumulator_t(tdims);
    acc0.clear(5);
    acc0.update(make_tensor<scalar_t>(tdims, -0.10).array(), 0);
    acc0.update(make_tensor<scalar_t>(tdims, -0.11).array(), 0);
    acc0.update(make_tensor<scalar_t>(tdims, -0.12).array(), 0);
    acc0.update(make_tensor<scalar_t>(tdims, -0.09).array(), 0);
    acc0.update(make_tensor<scalar_t>(tdims, -0.08).array(), 0);
    acc0.update(make_tensor<scalar_t>(tdims, -0.20).array(), 1);
    acc0.update(make_tensor<scalar_t>(tdims, -0.30).array(), 1);
    acc0.update(make_tensor<scalar_t>(tdims, -0.40).array(), 1);
    acc0.update(make_tensor<scalar_t>(tdims, -1.00).array(), 2);
    acc0.update(make_tensor<scalar_t>(tdims, -1.00).array(), 2);
    acc0.update(make_tensor<scalar_t>(tdims, -1.01).array(), 3);
    acc0.update(make_tensor<scalar_t>(tdims, -1.01).array(), 3);
    acc0.update(make_tensor<scalar_t>(tdims, -1.02).array(), 3);
    acc0.update(make_tensor<scalar_t>(tdims, -2.00).array(), 4);

    UTEST_CHECK_CLOSE(acc0.x0(0), 5.0, 1e-12);
    UTEST_CHECK_CLOSE(acc0.x0(1), 3.0, 1e-12);
    UTEST_CHECK_CLOSE(acc0.x0(2), 2.0, 1e-12);
    UTEST_CHECK_CLOSE(acc0.x0(3), 3.0, 1e-12);
    UTEST_CHECK_CLOSE(acc0.x0(4), 1.0, 1e-12);

    UTEST_CHECK_CLOSE(acc0.r1(0)(0), 0.5, 1e-12);
    UTEST_CHECK_CLOSE(acc0.r1(1)(0), 0.9, 1e-12);
    UTEST_CHECK_CLOSE(acc0.r1(2)(0), 2.0, 1e-12);
    UTEST_CHECK_CLOSE(acc0.r1(3)(0), 3.04, 1e-12);
    UTEST_CHECK_CLOSE(acc0.r1(4)(0), 2.0, 1e-12);

    UTEST_CHECK_CLOSE(acc0.r2(0)(0), 0.0510, 1e-12);
    UTEST_CHECK_CLOSE(acc0.r2(1)(0), 0.29, 1e-12);
    UTEST_CHECK_CLOSE(acc0.r2(2)(0), 2.0, 1e-12);
    UTEST_CHECK_CLOSE(acc0.r2(3)(0), 3.0806, 1e-12);
    UTEST_CHECK_CLOSE(acc0.r2(4)(0), 4.0, 1e-12);

    return acc0;
}
} // namespace

UTEST_BEGIN_MODULE(test_wlearner_util)

UTEST_CASE(scale)
{
    const auto tables0 = make_tensor<scalar_t>(make_dims(4, 1, 1, 3), 1, 1, 1, 2, 3, 3, 3, 4, 5, 4, 4, 4);

    {
        const auto scale    = make_vector<scalar_t>(7.0);
        const auto expected = make_tensor<scalar_t>(tables0.dims(), 7, 7, 7, 14, 21, 21, 21, 28, 35, 28, 28, 28);

        auto tables = tables0;
        UTEST_REQUIRE_NOTHROW(wlearner::scale(tables, scale));
        UTEST_CHECK_CLOSE(tables, expected, 1e-15);
    }
    {
        const auto scale    = make_vector<scalar_t>(5.0, 7.0, 3.0, 2.0);
        const auto expected = make_tensor<scalar_t>(tables0.dims(), 5, 5, 5, 14, 21, 21, 9, 12, 15, 8, 8, 8);

        auto tables = tables0;
        UTEST_REQUIRE_NOTHROW(wlearner::scale(tables, scale));
        UTEST_CHECK_CLOSE(tables, expected, 1e-15);
    }
}

UTEST_CASE(reduce)
{
    std::vector<cache_t> caches;
    caches.emplace_back(1.0, tensor_size_t{0});
    caches.emplace_back(0.0, tensor_size_t{1});
    caches.emplace_back(2.0, tensor_size_t{2});
    caches.emplace_back(5.0, tensor_size_t{3});

    const auto& min = min_reduce(caches);
    UTEST_CHECK_EQUAL(min.m_index, 1);
    UTEST_CHECK_CLOSE(min.m_score, 0.0, 1e-12);

    const auto& sum = sum_reduce(caches, 10);
    UTEST_CHECK_EQUAL(sum.m_index, 0);
    UTEST_CHECK_CLOSE(sum.m_score, 0.8, 1e-12);
}

UTEST_CASE(accumulator)
{
    const auto tdims = make_dims(3, 1, 1);

    auto acc = wlearner::accumulator_t(tdims);
    acc.clear(2);

    UTEST_CHECK_EQUAL(acc.bins(), 2);
    UTEST_CHECK_EQUAL(acc.tdims(), tdims);

    {
        const auto expected = make_full_tensor<scalar_t>(tdims, 0.0);
        UTEST_CHECK_CLOSE(acc.fit_constant(0), expected.array(), 1e-12);
        UTEST_CHECK_CLOSE(acc.fit_constant(1), expected.array(), 1e-12);
    }
    UTEST_CHECK_CLOSE(acc.rss_zero(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.rss_zero(1), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.rss_constant(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.rss_constant(1), 0.0, 1e-12);

    tensor4d_t vgrads(cat_dims(5, tdims));
    vgrads.tensor(0).full(+0.0);
    vgrads.tensor(1).full(+1.0);
    vgrads.tensor(2).full(+2.0);
    vgrads.tensor(3).full(+3.0);
    vgrads.tensor(4).full(+4.0);

    acc.update(-2.0, vgrads.array(0), 0);
    acc.update(+2.0, vgrads.array(0), 1);

    acc.update(-1.0, vgrads.array(1), 0);
    acc.update(+1.0, vgrads.array(1), 1);

    acc.update(-3.0, vgrads.array(2), 0);
    acc.update(+3.0, vgrads.array(2), 1);

    acc.update(-4.0, vgrads.array(3), 0);
    acc.update(+4.0, vgrads.array(3), 1);

    acc.update(-1.0, vgrads.array(4), 1);
    acc.update(+1.0, vgrads.array(4), 1);

    UTEST_CHECK_CLOSE(acc.x0(0), +4.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.x0(1), +6.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.x1(0), -10.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.x1(1), +10.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.x2(0), +30.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.x2(1), +32.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.r1(0).minCoeff(), -6.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.r1(0).maxCoeff(), -6.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.r1(1).minCoeff(), -14.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.r1(1).maxCoeff(), -14.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.rx(0).minCoeff(), +19.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.rx(0).maxCoeff(), +19.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.rx(1).minCoeff(), -19.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.rx(1).maxCoeff(), -19.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.r2(0).minCoeff(), +14.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.r2(0).maxCoeff(), +14.0, 1e-12);

    UTEST_CHECK_CLOSE(acc.r2(1).minCoeff(), +46.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.r2(1).maxCoeff(), +46.0, 1e-12);

    {
        const auto expected0 = make_full_tensor<scalar_t>(tdims, -6.0 / 4.0);
        const auto expected1 = make_full_tensor<scalar_t>(tdims, -14.0 / 6.0);
        UTEST_CHECK_CLOSE(acc.fit_constant(0), expected0.array(), 1e-12);
        UTEST_CHECK_CLOSE(acc.fit_constant(1), expected1.array(), 1e-12);
    }
    UTEST_CHECK_CLOSE(acc.rss_zero(0), 42.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.rss_zero(1), 138.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.rss_constant(0), 15.0, 1e-12);
    UTEST_CHECK_CLOSE(acc.rss_constant(1), 40.0, 1e-12);
}

UTEST_CASE(accumulator_order)
{
    const auto acc = make_accumulator();
    const auto map = acc.sort();

    UTEST_REQUIRE_EQUAL(map.size(), 5U);

    UTEST_CHECK_CLOSE(map[0U].first, -square(2.0) / 1.0, 1e-12);
    UTEST_CHECK_CLOSE(map[1U].first, -square(1.01 + 1.01 + 1.02) / 3.0, 1e-12);
    UTEST_CHECK_CLOSE(map[2U].first, -square(1.00 + 1.00) / 2.00, 1e-12);
    UTEST_CHECK_CLOSE(map[3U].first, -square(0.20 + 0.30 + 0.40) / 3.0, 1e-12);
    UTEST_CHECK_CLOSE(map[4U].first, -square(0.08 + 0.09 + 0.10 + 0.11 + 0.12) / 5.0, 1e-12);

    UTEST_CHECK_EQUAL(map[0U].second, 4);
    UTEST_CHECK_EQUAL(map[1U].second, 3);
    UTEST_CHECK_EQUAL(map[2U].second, 2);
    UTEST_CHECK_EQUAL(map[3U].second, 1);
    UTEST_CHECK_EQUAL(map[4U].second, 0);
}

UTEST_CASE(accumulator_cluster)
{
    const auto acc                                                          = make_accumulator();
    const auto [cluster_x0, cluster_r1, cluster_r2, cluster_rx, cluster_id] = acc.cluster();

    UTEST_CHECK_CLOSE(cluster_x0,
                      make_tensor<scalar_t>(make_dims(5, 5), 5, 3, 2, 3, 1, 5, 3, 5, 1, 1, 8, 5, 1, 1, 1, 13, 1, 1, 1,
                                            1, 14, 1, 1, 1, 1),
                      1e-12);

    UTEST_CHECK_CLOSE(cluster_r1,
                      make_tensor<scalar_t>(make_dims(5, 5, 1, 1, 1), 0.5, 0.9, 2, 3.04, 2, 0.5, 0.9, 5.04, 2, 2, 1.4,
                                            5.04, 2, 2, 2, 6.44, 2, 2, 2, 2, 8.44, 2, 2, 2, 2),
                      1e-12);

    UTEST_CHECK_CLOSE(cluster_r2,
                      make_tensor<scalar_t>(make_dims(5, 5, 1, 1, 1), 0.051, 0.29, 2, 3.0806, 4, 0.051, 0.29, 5.0806, 4,
                                            4, 0.341, 5.0806, 4, 4, 4, 5.4216, 4, 4, 4, 4, 9.4216, 4, 4, 4, 4),
                      1e-12);

    UTEST_CHECK_CLOSE(cluster_rx,
                      make_tensor<scalar_t>(make_dims(5, 5, 1, 1, 1), 0.1, 0.3, 1.0, 3.04 / 3.0, 2, 0.1, 0.3, 1.008, 2,
                                            2, 0.175, 1.008, 2, 2, 2, 6.44 / 13.0, 2, 2, 2, 2, 8.44 / 14.0, 2, 2, 2, 2),
                      1e-12);

    UTEST_CHECK_EQUAL(cluster_id, make_tensor<tensor_size_t>(make_dims(5, 5), 0, 1, 2, 3, 4, 0, 1, 2, 2, 3, 0, 0, 1, 1,
                                                             2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0));
}

UTEST_CASE(criterion)
{
    const auto rss = std::numbers::e;
    const auto n   = 100;
    const auto k   = 3;

    UTEST_CHECK_CLOSE(wlearner::make_score(wlearner_criterion::rss, rss, k, n), rss, 1e-12);
    UTEST_CHECK_CLOSE(wlearner::make_score(wlearner_criterion::aic, rss, k, n), -354.517018598809136804, 1e-12);
    UTEST_CHECK_CLOSE(wlearner::make_score(wlearner_criterion::aicc, rss, k, n), -354.267018598809136804, 1e-12);
    UTEST_CHECK_CLOSE(wlearner::make_score(wlearner_criterion::bic, rss, k, n), -346.70150804084486269988, 1e-12);
}

UTEST_END_MODULE()
