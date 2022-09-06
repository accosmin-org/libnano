#include <nano/datasource.h>
#include <utest/utest.h>

using namespace nano;

template <typename tscalar, size_t trank>
static auto check_inputs(const datasource_t& datasource, tensor_size_t index, const feature_t& gt_feature,
                         const tensor_mem_t<tscalar, trank>& gt_data, const mask_cmap_t& gt_mask)
{
    const auto visitor = [&](const auto& feature, const auto& data, const auto& mask)
    {
        UTEST_CHECK_EQUAL(feature, gt_feature);
        if constexpr (std::is_same<decltype(data), const tensor_cmap_t<tscalar, trank>&>::value)
        {
            UTEST_CHECK_CLOSE(data, gt_data, 1e-12);
            UTEST_CHECK_EQUAL(mask, gt_mask);
        }
        else
        {
            UTEST_CHECK(false);
        }
    };

    datasource.visit_inputs(index, visitor);
}

template <typename tscalar, size_t trank>
static auto check_target(const datasource_t& datasource, const feature_t& gt_feature,
                         const tensor_mem_t<tscalar, trank>& gt_data, const mask_cmap_t& gt_mask)
{
    const auto visitor = [&](const auto& feature, const auto& data, const auto& mask)
    {
        UTEST_CHECK_EQUAL(feature, gt_feature);
        if constexpr (std::is_same<decltype(data), const tensor_cmap_t<tscalar, trank>&>::value)
        {
            UTEST_CHECK_CLOSE(data, gt_data, 1e-12);
            UTEST_CHECK_EQUAL(mask, gt_mask);
        }
        else
        {
            UTEST_CHECK(false);
        }
    };

    datasource.visit_target(visitor);
}
