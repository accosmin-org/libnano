#include <utest/utest.h>
#include <nano/numeric.h>
#include "fixture/gboost.h"

using namespace nano;

class whinge_dataset_t : public fixture_dataset_t
{
public:

    whinge_dataset_t() = default;

    [[nodiscard]] tensor_size_t groups() const override
    {
        return 1;
    }

    void check_wlearner(const wlearner_hinge_t& wlearner) const
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), feature());
        UTEST_CHECK_EQUAL(wlearner.negative(), negative());
        UTEST_CHECK_EQUAL(wlearner.tables().dims(), tables().dims());
        UTEST_CHECK_CLOSE(wlearner.threshold(), threshold(), 1e-8);
        UTEST_CHECK_EIGEN_CLOSE(wlearner.tables().array(), tables().array(), 1e-8);
    }

    [[nodiscard]] scalar_t threshold() const { return 2.5; }
    [[nodiscard]] tensor_size_t feature(bool discrete = false) const { return get_feature(discrete); }
    [[nodiscard]] virtual bool negative() const = 0;
    [[nodiscard]] virtual tensor4d_t tables() const = 0;
};

class whinge_neg_dataset_t : public whinge_dataset_t
{
public:

    whinge_neg_dataset_t() = default;

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(
            make_hinge_target(sample, feature(), 5, 2.5, +3.0, 0.0, 0));
    }

    [[nodiscard]] bool negative() const override
    {
        return true;
    }

    [[nodiscard]] tensor4d_t tables() const override
    {
        return {make_dims(2, 1, 1, 1), {+3.0, -3.0 * threshold()}};
    }
};

class whinge_pos_dataset_t : public whinge_dataset_t
{
public:

    whinge_pos_dataset_t() = default;

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(
            make_hinge_target(sample, feature(), 5, 2.5, 0.0, -2.1, 0));
    }

    [[nodiscard]] bool negative() const override
    {
        return false;
    }

    [[nodiscard]] tensor4d_t tables() const override
    {
        return {make_dims(2, 1, 1, 1), {-2.1, +2.1 * threshold()}};
    }
};

UTEST_BEGIN_MODULE(test_gboost_whinge)

UTEST_CASE(fitting_neg)
{
    const auto dataset = make_dataset<whinge_neg_dataset_t>();
    const auto datasetx1 = make_dataset<whinge_neg_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<whinge_neg_dataset_t>(dataset.feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_continuous_features_dataset_t<whinge_neg_dataset_t>>();

    auto wlearner = make_wlearner<wlearner_hinge_t>();
    check_no_fit(wlearner, datasetx3);
    check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3);
}

UTEST_CASE(fitting_pos)
{
    const auto dataset = make_dataset<whinge_pos_dataset_t>();
    const auto datasetx1 = make_dataset<whinge_pos_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<whinge_pos_dataset_t>(dataset.feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_continuous_features_dataset_t<whinge_pos_dataset_t>>();

    auto wlearner = make_wlearner<wlearner_hinge_t>();
    check_no_fit(wlearner, datasetx3);
    check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3);
}

UTEST_END_MODULE()
