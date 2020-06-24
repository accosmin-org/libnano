#include <utest/utest.h>
#include <nano/numeric.h>
#include <nano/dataset/synth_affine.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_synthetic)

UTEST_CASE(affine)
{
    auto dataset = synthetic_affine_dataset_t{};

    dataset.folds(3);
    dataset.noise(0);
    dataset.modulo(2);
    dataset.samples(100);
    dataset.train_percentage(50);
    dataset.idim(make_dims(7, 1, 1));
    dataset.tdim(make_dims(3, 1, 1));

    UTEST_REQUIRE(dataset.load());

    const auto tfeature = dataset.tfeature();
    UTEST_CHECK(!tfeature.discrete());
    UTEST_CHECK(!tfeature.optional());

    const auto& bias = dataset.bias();
    UTEST_REQUIRE_EQUAL(bias.size(), 3);

    const auto& weights = dataset.weights();
    UTEST_REQUIRE_EQUAL(weights.rows(), 7);
    UTEST_REQUIRE_EQUAL(weights.cols(), 3);
    for (tensor_size_t row = 0; row < weights.rows(); ++ row)
    {
        if ((row % dataset.modulo()) > 0)
        {
            UTEST_CHECK_EIGEN_CLOSE(weights.row(row).transpose(), vector_t::Zero(weights.cols()), epsilon1<scalar_t>());
        }
    }

    UTEST_CHECK_EQUAL(dataset.folds(), 3);
    UTEST_CHECK_EQUAL(dataset.samples(), 100);
    UTEST_CHECK_EQUAL(dataset.samples(fold_t{0, protocol::train}), 50);
    UTEST_CHECK_EQUAL(dataset.samples(fold_t{0, protocol::valid}), 25);
    UTEST_CHECK_EQUAL(dataset.samples(fold_t{0, protocol::test}), 25);

    for (size_t f = 0; f < dataset.folds(); ++ f)
    {
        const auto tr_inputs = dataset.inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset.inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset.inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset.targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset.targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset.targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_inputs.dims(), nano::make_dims(50, 7, 1, 1));
        UTEST_CHECK_EQUAL(vd_inputs.dims(), nano::make_dims(25, 7, 1, 1));
        UTEST_CHECK_EQUAL(te_inputs.dims(), nano::make_dims(25, 7, 1, 1));

        UTEST_CHECK_EQUAL(tr_targets.dims(), nano::make_dims(50, 3, 1, 1));
        UTEST_CHECK_EQUAL(vd_targets.dims(), nano::make_dims(25, 3, 1, 1));
        UTEST_CHECK_EQUAL(te_targets.dims(), nano::make_dims(25, 3, 1, 1));

        for (tensor_size_t s = 0; s < 100; ++ s)
        {
            const auto row = (s < 50) ? s : (s < 75 ? (s - 50) : (s - 75));
            const auto& inputs = (s < 50) ? tr_inputs : (s < 75 ? vd_inputs : te_inputs);
            const auto& targets = (s < 50) ? tr_targets : (s < 75 ? vd_targets : te_targets);

            UTEST_CHECK_EIGEN_CLOSE(targets.vector(row), weights.transpose() * inputs.vector(row) + bias, epsilon1<scalar_t>());
        }
    }
}

UTEST_END_MODULE()
