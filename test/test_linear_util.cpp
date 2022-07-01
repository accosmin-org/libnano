#include "fixture/linear.h"
#include "fixture/loss.h"
#include <nano/linear/cache.h>
#include <nano/linear/util.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_linear_util)

UTEST_CASE(cache)
{
    const auto fill_cache = [](linear::cache_t& cache, scalar_t value)
    {
        cache.m_vm1 = value;
        cache.m_vm2 = value * value;
        cache.m_gb1.full(value);
        cache.m_gb2.full(value * value);
        cache.m_gW1.full(value);
        cache.m_gW2.full(value * value);
    };

    const auto make_caches = [&](bool g1, bool g2)
    {
        auto caches = std::vector<linear::cache_t>(3U, linear::cache_t{3, 2, g1, g2});
        fill_cache(caches[0], 1);
        fill_cache(caches[1], 2);
        fill_cache(caches[2], 3);
        return caches;
    };

    {
        auto caches = make_caches(false, false);

        const auto& cache0 = linear::cache_t::reduce(caches, 6);
        UTEST_CHECK_CLOSE(cache0.m_vm1, 6.0 / 6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_vm2, 14.0 / 6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb1, make_full_tensor<scalar_t>(make_dims(0), 6.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb2, make_full_tensor<scalar_t>(make_dims(0), 14.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW1, make_full_tensor<scalar_t>(make_dims(0, 0), 6.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW2, make_full_tensor<scalar_t>(make_dims(0, 0), 14.0 / 6.0), 1e-12);
    }
    {
        auto caches = make_caches(false, true);

        const auto& cache0 = linear::cache_t::reduce(caches, 6);
        UTEST_CHECK_CLOSE(cache0.m_vm1, 6.0 / 6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_vm2, 14.0 / 6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb1, make_full_tensor<scalar_t>(make_dims(0), 6.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb2, make_full_tensor<scalar_t>(make_dims(0), 14.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW1, make_full_tensor<scalar_t>(make_dims(0, 0), 6.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW2, make_full_tensor<scalar_t>(make_dims(0, 0), 14.0 / 6.0), 1e-12);
    }
    {
        auto caches = make_caches(true, false);

        const auto& cache0 = linear::cache_t::reduce(caches, 6);
        UTEST_CHECK_CLOSE(cache0.m_vm1, 6.0 / 6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_vm2, 14.0 / 6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb1, make_full_tensor<scalar_t>(make_dims(2), 6.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb2, make_full_tensor<scalar_t>(make_dims(0), 14.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW1, make_full_tensor<scalar_t>(make_dims(2, 3), 6.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW2, make_full_tensor<scalar_t>(make_dims(0, 0), 14.0 / 6.0), 1e-12);
    }
    {
        auto caches = make_caches(true, true);

        const auto& cache0 = linear::cache_t::reduce(caches, 6);
        UTEST_CHECK_CLOSE(cache0.m_vm1, 6.0 / 6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_vm2, 14.0 / 6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb1, make_full_tensor<scalar_t>(make_dims(2), 6.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb2, make_full_tensor<scalar_t>(make_dims(2), 14.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW1, make_full_tensor<scalar_t>(make_dims(2, 3), 6.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW2, make_full_tensor<scalar_t>(make_dims(2, 3), 14.0 / 6.0), 1e-12);
    }
}

UTEST_CASE(predict)
{
    const auto epsilon = epsilon1<scalar_t>();

    tensor1d_t bias(3);
    bias.random();
    tensor2d_t weights(3, 5);
    weights.random();
    tensor2d_t inputs(11, 5);
    inputs.random();

    tensor4d_t outputs;
    linear::predict(inputs, weights, bias, outputs);

    for (tensor_size_t sample = 0; sample < inputs.size<0>(); ++sample)
    {
        UTEST_CHECK_CLOSE(outputs.vector(sample), weights.matrix() * inputs.vector(sample) + bias.vector(), epsilon);
    }
}

UTEST_END_MODULE()
