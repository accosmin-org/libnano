#include "fixture/utils.h"
#include "fixture/linear.h"
#include <nano/core/numeric.h>
#include <nano/linear/util.h>
#include <nano/linear/cache.h>
//#include <nano/linear/model.h>
//#include <nano/linear/function.h>

using namespace nano;

/*static auto make_samples()
{
    return arange(0, 100);
}

static auto make_dataset(const tensor_size_t isize = 5, const tensor_size_t tsize = 3)
{
    auto dataset = synthetic_affine_dataset_t{};
    dataset.noise(epsilon1<scalar_t>());
    dataset.idims(make_dims(isize, 1, 1));
    dataset.tdims(make_dims(tsize, 1, 1));
    dataset.modulo(1);
    dataset.samples(100);
    UTEST_CHECK_NOTHROW(dataset.load());
    return dataset;
}*/

UTEST_BEGIN_MODULE(test_linear)

UTEST_CASE(cache)
{
    const auto fill_cache = [] (linear::cache_t& cache, scalar_t value)
    {
        cache.m_vm1 = value;
        cache.m_vm2 = value * value;
        cache.m_gb1.full(value);
        cache.m_gb2.full(value * value);
        cache.m_gW1.full(value);
        cache.m_gW2.full(value * value);
    };

    const auto make_caches = [&] (bool g1, bool g2)
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
        UTEST_CHECK_CLOSE(cache0.m_vm1, 6.0/6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_vm2, 14.0/6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb1, make_full_tensor<scalar_t>(make_dims(0), 6.0/6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb2, make_full_tensor<scalar_t>(make_dims(0), 14.0/6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW1, make_full_tensor<scalar_t>(make_dims(0, 0), 6.0/6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW2, make_full_tensor<scalar_t>(make_dims(0, 0), 14.0/6.0), 1e-12);
    }
    {
        auto caches = make_caches(false, true);

        const auto& cache0 = linear::cache_t::reduce(caches, 6);
        UTEST_CHECK_CLOSE(cache0.m_vm1, 6.0/6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_vm2, 14.0/6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb1, make_full_tensor<scalar_t>(make_dims(0), 6.0/6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb2, make_full_tensor<scalar_t>(make_dims(0), 14.0/6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW1, make_full_tensor<scalar_t>(make_dims(0, 0), 6.0/6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW2, make_full_tensor<scalar_t>(make_dims(0, 0), 14.0/6.0), 1e-12);
    }
    {
        auto caches = make_caches(true, false);

        const auto& cache0 = linear::cache_t::reduce(caches, 6);
        UTEST_CHECK_CLOSE(cache0.m_vm1, 6.0/6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_vm2, 14.0/6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb1, make_full_tensor<scalar_t>(make_dims(2), 6.0/6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb2, make_full_tensor<scalar_t>(make_dims(0), 14.0/6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW1, make_full_tensor<scalar_t>(make_dims(3, 2), 6.0/6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW2, make_full_tensor<scalar_t>(make_dims(0, 0), 14.0/6.0), 1e-12);
    }
    {
        auto caches = make_caches(true, true);

        const auto& cache0 = linear::cache_t::reduce(caches, 6);
        UTEST_CHECK_CLOSE(cache0.m_vm1, 6.0/6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_vm2, 14.0/6.0, 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb1, make_full_tensor<scalar_t>(make_dims(2), 6.0/6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gb2, make_full_tensor<scalar_t>(make_dims(2), 14.0/6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW1, make_full_tensor<scalar_t>(make_dims(3, 2), 6.0/6.0), 1e-12);
        UTEST_CHECK_CLOSE(cache0.m_gW2, make_full_tensor<scalar_t>(make_dims(3, 2), 14.0/6.0), 1e-12);
    }
}

UTEST_CASE(predict)
{
    tensor1d_t bias(3); bias.random();
    tensor2d_t weights(5, 3); weights.random();
    tensor4d_t inputs(11, 5, 1, 1); inputs.random();

    tensor4d_t outputs;
    linear::predict(inputs, weights, bias, outputs);

    for (tensor_size_t sample = 0; sample < inputs.size<0>(); ++ sample)
    {
        UTEST_CHECK_CLOSE(
            outputs.vector(sample),
            weights.matrix().transpose() * inputs.vector(sample) + bias.vector(),
            epsilon1<scalar_t>());
    }
}

UTEST_CASE(dataset)
{
    const auto targets = tensor_size_t{3};
    const auto samples = tensor_size_t{100};
    const auto features = tensor_size_t{4};
    const auto epsilon = epsilon1<scalar_t>();

    auto dataset = fixture_dataset_t{};

    dataset.noise(0);
    dataset.modulo(31);
    dataset.samples(samples);
    dataset.targets(targets);
    dataset.features(features);

    UTEST_REQUIRE_NOTHROW(dataset.load());

    auto generator = dataset_generator_t{dataset};
    generator.add<elemwise_generator_t<sclass_identity_t>>();
    generator.add<elemwise_generator_t<mclass_identity_t>>();
    generator.add<elemwise_generator_t<scalar_identity_t>>();
    generator.add<elemwise_generator_t<struct_identity_t>>();

    UTEST_CHECK_EQUAL(generator.target(), feature_t{"Wx+b+eps"}.scalar(feature_type::float64, make_dims(targets, 1, 1)));

    const auto bias = dataset.bias().vector();
    UTEST_REQUIRE_EQUAL(bias.size(), targets);

    const auto weights = dataset.weights().matrix();
    UTEST_REQUIRE_EQUAL(weights.rows(), targets);
    UTEST_REQUIRE_EQUAL(weights.cols(), 14 * features / 4);

    UTEST_CHECK_EQUAL(dataset.features(), features);
    UTEST_CHECK_EQUAL(dataset.samples(), samples);
    UTEST_CHECK_EQUAL(dataset.test_samples(), arange(0, 0));
    UTEST_CHECK_EQUAL(dataset.train_samples(), arange(0, samples));

    auto called = make_full_tensor<tensor_size_t>(make_dims(samples), 0);

    auto iterator = flatten_iterator_t{generator, arange(0, samples)};
    iterator.exec(execution::seq);
    iterator.batch(100);
    iterator.loop([&] (tensor_range_t range, size_t, tensor2d_cmap_t inputs, tensor4d_cmap_t targets)
    {
        for (tensor_size_t i = 0, size = range.size(); i < size; ++ i)
        {
            UTEST_CHECK_CLOSE(targets.vector(i), weights * inputs.vector(i) + bias, epsilon);
            called(range.begin() + i) = 1;
        }
    });

    UTEST_CHECK_EQUAL(called, make_full_tensor<tensor_size_t>(make_dims(samples), 1));
}

/*
UTEST_CASE(evaluate)
{
    const auto loss = make_loss();
    const auto dataset = make_dataset();
    const auto samples = make_samples();

    auto function = linear_function_t{*loss, dataset, samples};
    UTEST_REQUIRE_EQUAL(function.size(), 5 * 3 + 3);
    UTEST_REQUIRE_EQUAL(function.isize(), 5);
    UTEST_REQUIRE_EQUAL(function.tsize(), 3);
    UTEST_REQUIRE_NOTHROW(function.l1reg(0));
    UTEST_REQUIRE_NOTHROW(function.l2reg(0));
    UTEST_REQUIRE_NOTHROW(function.vAreg(0));

    const auto inputs = dataset.inputs(make_samples());
    const auto targets = dataset.targets(make_samples());

    tensor4d_t outputs;
    const vector_t x = vector_t::Random(function.size());
    linear::predict(inputs, function.weights(x), function.bias(x), outputs);

    tensor1d_t values;
    loss->value(targets, outputs, values);

    for (tensor_size_t batch = 1; batch <= 16; ++ batch)
    {
        UTEST_REQUIRE_NOTHROW(function.batch(batch));
        UTEST_CHECK_LESS(std::fabs(function.vgrad(x) - values.vector().mean()), epsilon1<scalar_t>());
    }
}

UTEST_CASE(gradient)
{
    const auto loss = make_loss();
    const auto dataset = make_dataset();
    const auto samples = make_samples();

    auto function = linear_function_t{*loss, dataset, samples};
    UTEST_REQUIRE_EQUAL(function.size(), 5 * 3 + 3);
    UTEST_REQUIRE_EQUAL(function.isize(), 5);
    UTEST_REQUIRE_EQUAL(function.tsize(), 3);
    UTEST_REQUIRE_THROW(function.l1reg(-1e+0), std::runtime_error);
    UTEST_REQUIRE_THROW(function.l1reg(+1e+9), std::runtime_error);
    UTEST_REQUIRE_THROW(function.l2reg(-1e+0), std::runtime_error);
    UTEST_REQUIRE_THROW(function.l2reg(+1e+9), std::runtime_error);
    UTEST_REQUIRE_THROW(function.vAreg(-1e+0), std::runtime_error);
    UTEST_REQUIRE_THROW(function.vAreg(+1e+9), std::runtime_error);
    UTEST_REQUIRE_NOTHROW(function.l1reg(1e-1));
    UTEST_REQUIRE_NOTHROW(function.l2reg(1e+1));
    UTEST_REQUIRE_NOTHROW(function.vAreg(5e-1));

    const vector_t x = vector_t::Random(function.size());

    for (const auto scaling : enum_values<feature_scaling>())
    {
        UTEST_REQUIRE_NOTHROW(function.scaling(scaling));
        UTEST_CHECK_LESS(function.grad_accuracy(x), 10 * epsilon2<scalar_t>());
    }
}

UTEST_CASE(minimize)
{
    const auto loss = make_loss();
    const auto solver = make_solver("cgd", epsilon3<scalar_t>());
    const auto dataset = make_dataset(3, 2);
    const auto samples = make_samples();

    auto function = linear_function_t{*loss, dataset, samples};
    UTEST_REQUIRE_EQUAL(function.size(), 3 * 2 + 2);
    UTEST_REQUIRE_NOTHROW(function.l1reg(0.0));
    UTEST_REQUIRE_NOTHROW(function.l2reg(0.0));
    UTEST_REQUIRE_NOTHROW(function.vAreg(0.0));

    const auto state = solver->minimize(function, vector_t::Zero(function.size()));
    UTEST_CHECK(state);
    UTEST_CHECK(state.converged(solver->epsilon()));

    UTEST_CHECK_CLOSE(function.bias(state.x).vector(), dataset.bias(), 1e+1 * solver->epsilon());
    UTEST_CHECK_CLOSE(function.weights(state.x).matrix(), dataset.weights(), 1e+1 * solver->epsilon());
}

UTEST_CASE(train)
{
    const auto loss = make_loss();
    const auto solver = make_solver("cgd", epsilon3<scalar_t>());
    const auto dataset = make_dataset(3, 2);
    const auto samples = make_samples();

    for (const auto scaling : enum_values<feature_scaling>())
    {
        auto model = linear_model_t{};
        UTEST_REQUIRE_NOTHROW(model.batch(16));
        switch (scaling)
        {
        case feature_scaling::mean:     UTEST_REQUIRE_NOTHROW(model.l1reg(1e-6)); break;
        case feature_scaling::minmax:   UTEST_REQUIRE_NOTHROW(model.l2reg(1e-6)); break;
        case feature_scaling::standard: UTEST_REQUIRE_NOTHROW(model.vAreg(1e-6)); break;
        default:                        break;
        }
        UTEST_REQUIRE_NOTHROW(model.scaling(scaling));

        tensor4d_t outputs;
        UTEST_REQUIRE_NOTHROW(model.fit(*loss, dataset, samples, *solver));
        UTEST_REQUIRE_NOTHROW(outputs = model.predict(dataset, samples));

        UTEST_CHECK_CLOSE(model.bias().vector(), dataset.bias(), 1e+2 * solver->epsilon());
        UTEST_CHECK_CLOSE(model.weights().matrix(), dataset.weights(), 1e+2 * solver->epsilon());

        const auto targets = dataset.targets(samples);
        UTEST_CHECK_EQUAL(outputs.dims(), targets.dims());
        UTEST_CHECK_CLOSE(outputs.vector(), targets.vector(), 1e+1 * solver->epsilon());

        UTEST_REQUIRE_NOTHROW(outputs = model.predict(dataset, samples));
        UTEST_CHECK_EQUAL(outputs.dims(), targets.dims());
        UTEST_CHECK_CLOSE(outputs.vector(), targets.vector(), 1e+1 * solver->epsilon());

        outputs.random(-1, +2);
        UTEST_REQUIRE_NOTHROW(outputs = model.predict(dataset, samples));
        UTEST_CHECK_CLOSE(targets.vector(), outputs.vector(), 1e+1 * solver->epsilon());

        string_t str;
        {
            std::ostringstream stream;
            UTEST_REQUIRE_NOTHROW(model.write(stream));
            str = stream.str();
        }
        {
            auto new_model = linear_model_t{};
            std::istringstream stream(str);
            UTEST_REQUIRE_NOTHROW(new_model.read(stream));
            UTEST_CHECK_CLOSE(new_model.bias().vector(), model.bias().vector(), epsilon0<scalar_t>());
            UTEST_CHECK_CLOSE(new_model.weights().matrix(), model.weights().matrix(), epsilon0<scalar_t>());
        }
    }
}*/

UTEST_END_MODULE()
