#include <utest/utest.h>
#include <nano/numeric.h>
#include "fixture/gboost.h"

using namespace nano;

class wtable_dataset_t : public fixture_dataset_t
{
public:

    wtable_dataset_t() = default;

    [[nodiscard]] tensor_size_t groups() const override
    {
        return 3;
    }

    void make_target(const tensor_size_t sample) override
    {
        target(sample).constant(
            make_table_target(sample, feature(), 3, 5.0, 0));
    }

    void check_wlearner(const wlearner_table_t& wlearner) const
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), feature());
        UTEST_CHECK_EQUAL(wlearner.tables().dims(), tables().dims());
        UTEST_CHECK_EIGEN_CLOSE(wlearner.tables().array(), tables().array(), 1e-8);
    }

    [[nodiscard]] tensor_size_t the_discrete_feature() const { return feature(); }
    [[nodiscard]] tensor_size_t feature(bool discrete = true) const { return get_feature(discrete); }
    [[nodiscard]] tensor4d_t tables() const { return {make_dims(3, 1, 1, 1), std::array<scalar_t, 3>{{-5.0, +0.0, +5.0}}}; }
};

UTEST_BEGIN_MODULE(test_gboost_wtable)

UTEST_CASE(fitting)
{
    const auto dataset = make_dataset<wtable_dataset_t>();
    const auto datasetx1 = make_dataset<wtable_dataset_t>(dataset.isize(), dataset.tsize() + 1);
    const auto datasetx2 = make_dataset<wtable_dataset_t>(dataset.feature(), dataset.tsize());
    const auto datasetx3 = make_dataset<no_discrete_features_dataset_t<wtable_dataset_t>>();
    const auto datasetx4 = make_dataset<different_discrete_feature_dataset_t<wtable_dataset_t>>();

    auto wlearner = make_wlearner<wlearner_table_t>();
    check_no_fit(wlearner, datasetx3);
    check_wlearner(wlearner, dataset, datasetx1, datasetx2, datasetx3, datasetx4);
}

UTEST_END_MODULE()
