#include <utest/utest.h>
#include <nano/dataset/shuffle.h>
#include <nano/dataset/memfixed.h>

using namespace nano;

class Fixture final : public memfixed_dataset_t<uint8_t>
{
public:

    using memfixed_dataset_t<uint8_t>::resize;

    bool load() override
    {
        for (tensor_size_t s = 0; s < samples(); ++ s)
        {
            auto&& input = this->input(s);
            for (tensor_size_t f = 0; f < features(); ++ f)
            {
                input(f) = value(s, f);
            }
            target(s).constant(-s);
        }

        for (size_t f = 0; f < folds(); ++ f)
        {
            auto& split = this->split(f);

            const tensor_size_t tr_begin = 0, tr_end = tr_begin + samples() * 60 / 100;
            const tensor_size_t vd_begin = tr_end, vd_end = vd_begin + samples() * 30 / 100;
            const tensor_size_t te_begin = vd_end, te_end = samples();

            split.indices(protocol::train) = arange(tr_begin, tr_end);
            split.indices(protocol::valid) = arange(vd_begin, vd_end);
            split.indices(protocol::test) = arange(te_begin, te_end);

            UTEST_CHECK(split.valid(samples()));
        }

        return true;
    }

    static uint8_t value(tensor_size_t sample, tensor_size_t feature)
    {
        return static_cast<uint8_t>((sample + feature) % 256);
    }

    [[nodiscard]] feature_t tfeature() const override { return feature_t{"fixture"}; }
};

UTEST_BEGIN_MODULE(test_memfixed)

UTEST_CASE(load)
{
    auto dataset = Fixture{};

    dataset.folds(3);
    dataset.resize(nano::make_dims(100, 3, 10, 10), nano::make_dims(100, 10, 1, 1));
    UTEST_REQUIRE(dataset.load());

    UTEST_CHECK_EQUAL(dataset.folds(), 3);
    UTEST_CHECK_EQUAL(dataset.samples(), 100);
    UTEST_CHECK_EQUAL(dataset.samples(fold_t{0, protocol::train}), 60);
    UTEST_CHECK_EQUAL(dataset.samples(fold_t{0, protocol::valid}), 30);
    UTEST_CHECK_EQUAL(dataset.samples(fold_t{0, protocol::test}), 10);

    UTEST_CHECK_EQUAL(dataset.ifeature(0), feature_t{"feature_0_0_0"});
    UTEST_CHECK_EQUAL(dataset.ifeature(31), feature_t{"feature_0_3_1"});
    UTEST_CHECK_EQUAL(dataset.ifeature(257), feature_t{"feature_2_5_7"});
    UTEST_CHECK_EQUAL(dataset.tfeature(), feature_t{"fixture"});

    for (size_t f = 0; f < dataset.folds(); ++ f)
    {
        const auto tr_inputs = dataset.inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset.inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset.inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset.targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset.targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset.targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_inputs.dims(), nano::make_dims(60, 3, 10, 10));
        UTEST_CHECK_EQUAL(vd_inputs.dims(), nano::make_dims(30, 3, 10, 10));
        UTEST_CHECK_EQUAL(te_inputs.dims(), nano::make_dims(10, 3, 10, 10));

        UTEST_CHECK_EQUAL(tr_targets.dims(), nano::make_dims(60, 10, 1, 1));
        UTEST_CHECK_EQUAL(vd_targets.dims(), nano::make_dims(30, 10, 1, 1));
        UTEST_CHECK_EQUAL(te_targets.dims(), nano::make_dims(10, 10, 1, 1));

        for (tensor_size_t s = 0; s < 100; ++ s)
        {
            const auto row = (s < 60) ? s : (s < 90 ? (s - 60) : (s - 90));
            const auto& inputs = (s < 60) ? tr_inputs : (s < 90 ? vd_inputs : te_inputs);
            const auto& targets = (s < 60) ? tr_targets : (s < 90 ? vd_targets : te_targets);

            const auto imatrix = inputs.reshape(inputs.size<0>(), -1);
            for (tensor_size_t f = 0; f < 300; ++ f)
            {
                UTEST_CHECK_EQUAL(imatrix(row, f), Fixture::value(s, f));
            }

            UTEST_CHECK_CLOSE(targets.vector(row).minCoeff(), -s, 1e-8);
            UTEST_CHECK_CLOSE(targets.vector(row).maxCoeff(), -s, 1e-8);
        }
    }
}

UTEST_CASE(loop)
{
    auto dataset = Fixture{};

    dataset.folds(1);
    dataset.resize(nano::make_dims(100, 3, 16, 16), nano::make_dims(100, 10, 1, 1));
    UTEST_REQUIRE(dataset.load());

    for (const auto& fold : {fold_t{0U, protocol::test}})
    {
        for (const auto policy : {execution::seq, execution::par})
        {
            indices_t indices(dataset.samples(fold)); indices.constant(+0);
            indices_t threads(dataset.samples(fold)); threads.constant(-1);

            const tensor_size_t batch = 11;

            dataset.loop(policy, fold, batch, [&] (tensor_range_t range, size_t tnum)
            {
                UTEST_REQUIRE_LESS_EQUAL(0, range.begin());
                UTEST_REQUIRE_LESS(range.begin(), range.end());
                UTEST_REQUIRE_LESS_EQUAL(range.begin(), indices.size());
                UTEST_REQUIRE_LESS_EQUAL(range.size(), batch);
                UTEST_REQUIRE_LESS_EQUAL(0U, tnum);
                UTEST_REQUIRE_LESS(tnum, tpool_t::size());

                threads.slice(range).constant(tnum);

                const auto inputs = dataset.inputs(fold, range);
                UTEST_CHECK_EQUAL(inputs.template size<0>(), range.size());
                UTEST_CHECK_EQUAL(inputs.size(), range.size() * 3 * 16 * 16);

                const auto targets = dataset.targets(fold, range);
                UTEST_CHECK_EQUAL(targets.template size<0>(), range.size());
                UTEST_CHECK_EQUAL(targets.size(), range.size() * 10 * 1 * 1);

                UTEST_REQUIRE_EQUAL(0, indices.slice(range).vector().sum());
                UTEST_REQUIRE_EQUAL(0, indices.slice(range).min());
                UTEST_REQUIRE_EQUAL(0, indices.slice(range).max());
                indices.slice(range).constant(1);
                threads.slice(range).constant(static_cast<tensor_size_t>(tnum));
            });

            const auto max_threads = std::min(
                (dataset.samples(fold) + batch - 1) / batch,
                static_cast<tensor_size_t>(tpool_t::size()));

            UTEST_CHECK_EQUAL(indices.min(), 1);
            UTEST_CHECK_EQUAL(indices.max(), 1);
            UTEST_CHECK_EQUAL(indices.vector().sum(), indices.size());
            UTEST_CHECK_EQUAL(threads.min(), 0);
            UTEST_CHECK_LESS(threads.max(), max_threads);
        }
    }
}

UTEST_CASE(stats)
{
    auto dataset = Fixture{};

    dataset.folds(1);
    dataset.resize(nano::make_dims(100, 1, 2, 3), nano::make_dims(100, 10, 1, 1));
    UTEST_REQUIRE(dataset.load());

    const auto batch = 11;
    const auto istats = dataset.istats(fold_t{0U, protocol::train}, batch);

    UTEST_CHECK_EQUAL(istats.mean().template size<0>(), 1);
    UTEST_CHECK_EQUAL(istats.mean().template size<1>(), 2);
    UTEST_CHECK_EQUAL(istats.mean().template size<2>(), 3);

    UTEST_CHECK_EQUAL(istats.stdev().template size<0>(), 1);
    UTEST_CHECK_EQUAL(istats.stdev().template size<1>(), 2);
    UTEST_CHECK_EQUAL(istats.stdev().template size<2>(), 3);

    UTEST_CHECK_CLOSE(istats.min()(0), 0.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.min()(1), 1.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.min()(2), 2.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.min()(3), 3.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.min()(4), 4.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.min()(5), 5.0, 1e-8);

    UTEST_CHECK_CLOSE(istats.max()(0), 59.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.max()(1), 60.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.max()(2), 61.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.max()(3), 62.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.max()(4), 63.0, 1e-8);
    UTEST_CHECK_CLOSE(istats.max()(5), 64.0, 1e-8);

    UTEST_CHECK_CLOSE(istats.mean()(0), 29.5, 1e-8);
    UTEST_CHECK_CLOSE(istats.mean()(1), 30.5, 1e-8);
    UTEST_CHECK_CLOSE(istats.mean()(2), 31.5, 1e-8);
    UTEST_CHECK_CLOSE(istats.mean()(3), 32.5, 1e-8);
    UTEST_CHECK_CLOSE(istats.mean()(4), 33.5, 1e-8);
    UTEST_CHECK_CLOSE(istats.mean()(5), 34.5, 1e-8);

    UTEST_CHECK_CLOSE(istats.stdev().array().minCoeff(), 17.46425, 1e-6);
    UTEST_CHECK_CLOSE(istats.stdev().array().maxCoeff(), 17.46425, 1e-6);
}

UTEST_CASE(shuffle)
{
    auto dataset = Fixture{};

    dataset.folds(3);
    dataset.resize(nano::make_dims(100, 1, 8, 8), nano::make_dims(100, 3, 1, 1));
    UTEST_REQUIRE(dataset.load());

    const auto shuffled = shuffle_dataset_t{dataset, 13};

    UTEST_CHECK_EQUAL(shuffled.folds(), 3);
    UTEST_CHECK_EQUAL(shuffled.samples(), 100);
    UTEST_CHECK_EQUAL(shuffled.samples(fold_t{0, protocol::train}), 60);
    UTEST_CHECK_EQUAL(shuffled.samples(fold_t{0, protocol::valid}), 30);
    UTEST_CHECK_EQUAL(shuffled.samples(fold_t{0, protocol::test}), 10);

    UTEST_CHECK_EQUAL(shuffled.ifeature(0), feature_t{"feature_0_0_0"});
    UTEST_CHECK_EQUAL(shuffled.ifeature(31), feature_t{"feature_0_3_7"});
    UTEST_CHECK_EQUAL(shuffled.tfeature(), dataset.tfeature());

    const auto check_targets = [&] (const tensor4d_t& targets, tensor_range_t range)
    {
        UTEST_REQUIRE_EQUAL(targets.dims(), nano::make_dims(range.size(), 3, 1, 1));
        for (tensor_size_t s = range.begin(); s < range.end(); ++ s)
        {
            const auto row = s - range.begin();
            UTEST_CHECK_CLOSE(targets.vector(row).minCoeff(), -s, 1e-8);
            UTEST_CHECK_CLOSE(targets.vector(row).maxCoeff(), -s, 1e-8);
        }
    };

    const auto check_inputs = [&] (const auto& inputs, tensor_range_t range, const indices_t& features)
    {
        const auto imatrix = inputs.reshape(range.size(), -1);
        UTEST_REQUIRE_EQUAL(imatrix.cols(), features.size());
        for (tensor_size_t s = range.begin(); s < range.end(); ++ s)
        {
            const auto row = s - range.begin();
            for (tensor_size_t f = 0; f < features.size(); ++ f)
            {
                if (features(f) != 13)
                {
                    UTEST_CHECK_EQUAL(imatrix(row, f), Fixture::value(s, features(f)));
                }
            }
        }

        const auto* const it = std::find(begin(features), end(features), 13);
        if (it != features.end())
        {
            const auto f = std::distance(begin(features), it);

            tensor1d_t original(range.size());
            tensor1d_t permuted(range.size());
            for (tensor_size_t s = range.begin(); s < range.end(); ++ s)
            {
                const auto row = s - range.begin();
                original(row) = Fixture::value(s, 13);
                permuted(row) = imatrix(row, f);
            }

            UTEST_CHECK(std::is_permutation(begin(original), end(original), begin(permuted)));
        }
    };

    {
        const auto range = ::nano::make_range(0, 60);
        const auto targets = shuffled.targets(fold_t{0, protocol::train});
        check_targets(targets, range);
    }
    {
        const auto range = ::nano::make_range(11, 30);
        const auto targets = shuffled.targets(fold_t{0, protocol::train}, range);
        check_targets(targets, range);
    }
    {
        const auto range = ::nano::make_range(0, 60);
        const auto inputs = shuffled.inputs(fold_t{0, protocol::train});
        check_inputs(inputs, range, ::nano::arange(0, 64));
    }
    {
        const auto range = ::nano::make_range(17, 24);
        const auto inputs = shuffled.inputs(fold_t{0, protocol::train}, range);
        check_inputs(inputs, range, ::nano::arange(0, 64));
    }
    {
        const auto range = ::nano::make_range(17, 24);
        const auto inputs = shuffled.inputs(fold_t{0, protocol::train}, range, 22);
        check_inputs(inputs, range, std::array<tensor_size_t, 1>{{22}});
    }
    {
        const auto range = ::nano::make_range(17, 24);
        const auto inputs = shuffled.inputs(fold_t{0, protocol::train}, range, 13);
        check_inputs(inputs, range, std::array<tensor_size_t, 1>{{13}});
    }
    {
        const auto features = std::array<tensor_size_t, 3>{{1, 7, 14}};
        const auto range = ::nano::make_range(17, 24);
        const auto inputs = shuffled.inputs(fold_t{0, protocol::train}, range, features);
        check_inputs(inputs, range, features);
    }
    {
        const auto features = std::array<tensor_size_t, 3>{{1, 7, 13}};
        const auto range = ::nano::make_range(17, 24);
        const auto inputs = shuffled.inputs(fold_t{0, protocol::train}, range, features);
        check_inputs(inputs, range, features);
    }
    {
        const auto features = std::array<tensor_size_t, 3>{{13, 1, 7}};
        const auto range = ::nano::make_range(17, 24);
        const auto inputs = shuffled.inputs(fold_t{0, protocol::train}, range, features);
        check_inputs(inputs, range, features);
    }
}

UTEST_END_MODULE()
