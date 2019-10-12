#include <utest/utest.h>
#include <nano/memfixed.h>

using namespace nano;

class Fixture final : public memfixed_dataset_t<uint8_t>
{
public:

    using memfixed_dataset_t<uint8_t>::resize;

    bool load() override
    {
        for (tensor_size_t s = 0; s < samples(); ++ s)
        {
            input(s).vector().setConstant(static_cast<uint8_t>(s % 256));
            target(s).vector().setConstant(-s);
        }

        for (size_t f = 0; f < folds(); ++ f)
        {
            auto& split = this->split(f);

            const tensor_size_t tr_begin = 0, tr_end = tr_begin + samples() * 60 / 100;
            const tensor_size_t vd_begin = tr_end, vd_end = vd_begin + samples() * 30 / 100;
            const tensor_size_t te_begin = vd_end, te_end = samples();

            split.indices(protocol::train) = indices_t::LinSpaced(tr_end - tr_begin, tr_begin, tr_end);
            split.indices(protocol::valid) = indices_t::LinSpaced(vd_end - vd_begin, vd_begin, vd_end);
            split.indices(protocol::test) = indices_t::LinSpaced(te_end - te_begin, te_begin, te_end);

            UTEST_CHECK(split.valid(samples()));
        }

        return true;
    }

    feature_t tfeature() const override { return feature_t{"fixture"}; }
};

UTEST_BEGIN_MODULE(test_memfixed)

UTEST_CASE(load)
{
    auto dataset = Fixture{};

    dataset.folds(3);
    dataset.resize(nano::make_dims(100, 3, 16, 16), nano::make_dims(100, 10, 1, 1));
    UTEST_REQUIRE(dataset.load());

    UTEST_CHECK_EQUAL(dataset.folds(), 3);
    UTEST_CHECK_EQUAL(dataset.samples(), 100);
    UTEST_CHECK_EQUAL(dataset.samples(fold_t{0, protocol::train}), 60);
    UTEST_CHECK_EQUAL(dataset.samples(fold_t{0, protocol::valid}), 30);
    UTEST_CHECK_EQUAL(dataset.samples(fold_t{0, protocol::test}), 10);

    for (size_t f = 0; f < dataset.folds(); ++ f)
    {
        const auto tr_inputs = dataset.inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset.inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset.inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset.targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset.targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset.targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_inputs.dims(), nano::make_dims(60, 3, 16, 16));
        UTEST_CHECK_EQUAL(vd_inputs.dims(), nano::make_dims(30, 3, 16, 16));
        UTEST_CHECK_EQUAL(te_inputs.dims(), nano::make_dims(10, 3, 16, 16));

        UTEST_CHECK_EQUAL(tr_targets.dims(), nano::make_dims(60, 10, 1, 1));
        UTEST_CHECK_EQUAL(vd_targets.dims(), nano::make_dims(30, 10, 1, 1));
        UTEST_CHECK_EQUAL(te_targets.dims(), nano::make_dims(10, 10, 1, 1));

        for (tensor_size_t s = 0; s < 100; ++ s)
        {
            const auto row = (s < 60) ? s : (s < 90 ? (s - 60) : (s - 90));
            const auto& inputs = (s < 60) ? tr_inputs : (s < 90 ? vd_inputs : te_inputs);
            const auto& targets = (s < 60) ? tr_targets : (s < 90 ? vd_targets : te_targets);

            UTEST_CHECK_EQUAL(inputs.vector(row).minCoeff(), static_cast<uint8_t>(s % 256));
            UTEST_CHECK_EQUAL(inputs.vector(row).maxCoeff(), static_cast<uint8_t>(s % 256));

            UTEST_CHECK_CLOSE(targets.vector(row).minCoeff(), -s, 1e-8);
            UTEST_CHECK_CLOSE(targets.vector(row).maxCoeff(), -s, 1e-8);
        }
    }
}

UTEST_END_MODULE()
