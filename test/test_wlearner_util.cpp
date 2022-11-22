#include <nano/wlearner/accumulator.h>
#include <nano/wlearner/mhash.h>
#include <nano/wlearner/reduce.h>
#include <nano/wlearner/util.h>
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

    const auto& min = wlearner::min_reduce(caches);
    UTEST_CHECK_EQUAL(min.m_index, 1);
    UTEST_CHECK_CLOSE(min.m_score, 0.0, 1e-12);

    const auto& sum = wlearner::sum_reduce(caches, 10);
    UTEST_CHECK_EQUAL(sum.m_index, 0);
    UTEST_CHECK_CLOSE(sum.m_score, 0.8, 1e-12);
}

UTEST_CASE(accumulator)
{
    const auto tdims = make_dims(3, 1, 1);

    auto acc = wlearner::accumulator_t(tdims);
    acc.clear(2);

    UTEST_CHECK_EQUAL(acc.fvalues(), 2);
    UTEST_CHECK_EQUAL(acc.tdims(), tdims);

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
}

UTEST_CASE(mhash_hash)
{
    const auto fvalues = make_tensor<int8_t, 2>(make_dims(10, 3), 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                                0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1);

    const auto hash0 = ::nano::mhash(fvalues.array(0));
    const auto hash1 = ::nano::mhash(fvalues.array(1));
    const auto hash2 = ::nano::mhash(fvalues.array(2));
    const auto hash3 = ::nano::mhash(fvalues.array(3));
    const auto hash4 = ::nano::mhash(fvalues.array(4));
    const auto hash5 = ::nano::mhash(fvalues.array(5));
    const auto hash6 = ::nano::mhash(fvalues.array(6));
    const auto hash7 = ::nano::mhash(fvalues.array(7));
    const auto hash8 = ::nano::mhash(fvalues.array(8));
    const auto hash9 = ::nano::mhash(fvalues.array(9));

    UTEST_CHECK_EQUAL(hash4, hash0);
    UTEST_CHECK_EQUAL(hash5, hash6);
    UTEST_CHECK_EQUAL(hash7, hash1);
    UTEST_CHECK_EQUAL(hash8, hash3);

    UTEST_CHECK_NOT_EQUAL(hash0, hash1);
    UTEST_CHECK_NOT_EQUAL(hash0, hash2);
    UTEST_CHECK_NOT_EQUAL(hash0, hash3);
    UTEST_CHECK_NOT_EQUAL(hash0, hash5);
    UTEST_CHECK_NOT_EQUAL(hash0, hash6);
    UTEST_CHECK_NOT_EQUAL(hash0, hash7);
    UTEST_CHECK_NOT_EQUAL(hash0, hash8);
    UTEST_CHECK_NOT_EQUAL(hash0, hash9);

    UTEST_CHECK_NOT_EQUAL(hash1, hash2);
    UTEST_CHECK_NOT_EQUAL(hash1, hash3);
    UTEST_CHECK_NOT_EQUAL(hash1, hash4);
    UTEST_CHECK_NOT_EQUAL(hash1, hash6);
    UTEST_CHECK_NOT_EQUAL(hash1, hash8);
    UTEST_CHECK_NOT_EQUAL(hash1, hash9);

    UTEST_CHECK_NOT_EQUAL(hash2, hash3);
    UTEST_CHECK_NOT_EQUAL(hash2, hash4);
    UTEST_CHECK_NOT_EQUAL(hash2, hash5);
    UTEST_CHECK_NOT_EQUAL(hash2, hash6);
    UTEST_CHECK_NOT_EQUAL(hash2, hash7);
    UTEST_CHECK_NOT_EQUAL(hash2, hash8);
    UTEST_CHECK_NOT_EQUAL(hash2, hash9);

    UTEST_CHECK_NOT_EQUAL(hash3, hash4);
    UTEST_CHECK_NOT_EQUAL(hash3, hash5);
    UTEST_CHECK_NOT_EQUAL(hash3, hash6);
    UTEST_CHECK_NOT_EQUAL(hash3, hash7);
    UTEST_CHECK_NOT_EQUAL(hash3, hash9);

    UTEST_CHECK_NOT_EQUAL(hash4, hash5);
    UTEST_CHECK_NOT_EQUAL(hash4, hash6);
    UTEST_CHECK_NOT_EQUAL(hash4, hash7);
    UTEST_CHECK_NOT_EQUAL(hash4, hash8);
    UTEST_CHECK_NOT_EQUAL(hash4, hash9);

    UTEST_CHECK_NOT_EQUAL(hash5, hash7);
    UTEST_CHECK_NOT_EQUAL(hash5, hash8);
    UTEST_CHECK_NOT_EQUAL(hash5, hash9);

    UTEST_CHECK_NOT_EQUAL(hash6, hash7);
    UTEST_CHECK_NOT_EQUAL(hash6, hash8);
    UTEST_CHECK_NOT_EQUAL(hash6, hash9);

    UTEST_CHECK_NOT_EQUAL(hash7, hash8);
    UTEST_CHECK_NOT_EQUAL(hash7, hash9);

    UTEST_CHECK_NOT_EQUAL(hash8, hash9);
}

UTEST_CASE(mhash_make)
{
    const auto fvalues = make_tensor<int8_t, 2>(make_dims(12, 3), 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                                0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, -1, -1, -1, 0, 0, 0);

    const auto mhashes = make_mhashes(fvalues);
    UTEST_CHECK_EQUAL(mhashes.size(), 6);

    const auto fvalues_test =
        make_tensor<int8_t, 2>(make_dims(7, 3), -1, -1, -1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1);

    const auto index0 = ::nano::find(mhashes, fvalues_test.array(0));
    const auto index1 = ::nano::find(mhashes, fvalues_test.array(1));
    const auto index2 = ::nano::find(mhashes, fvalues_test.array(2));
    const auto index3 = ::nano::find(mhashes, fvalues_test.array(3));
    const auto index4 = ::nano::find(mhashes, fvalues_test.array(4));
    const auto index5 = ::nano::find(mhashes, fvalues_test.array(5));
    const auto index6 = ::nano::find(mhashes, fvalues_test.array(6));

    UTEST_CHECK_EQUAL(index0, -1);
    UTEST_CHECK_EQUAL(index1, +0);
    UTEST_CHECK_EQUAL(index2, +4);
    UTEST_CHECK_EQUAL(index3, +2);
    UTEST_CHECK_EQUAL(index4, +5);
    UTEST_CHECK_EQUAL(index5, +3);
    UTEST_CHECK_EQUAL(index6, +1);
}

UTEST_END_MODULE()
