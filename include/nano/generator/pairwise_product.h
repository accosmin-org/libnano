#pragma once

#include <nano/generator/pairwise.h>

namespace nano
{
///
/// \brief pairwise product of scalar features.
///
class NANO_PUBLIC pairwise_product_t : public pairwise_input_scalar_scalar_t, public generated_scalar_t
{
public:
    pairwise_product_t()
        : pairwise_input_scalar_scalar_t("product")
    {
    }

    feature_t feature(tensor_size_t ifeature) const override { return make_scalar_feature(ifeature, "product"); }

    static auto process(tensor_size_t)
    {
        const auto colsize = tensor_size_t{1};
        const auto process = [](const auto& values1, const auto& values2)
        { return static_cast<scalar_t>(values1(0)) * static_cast<scalar_t>(values2(0)); };

        return std::make_tuple(process, colsize);
    }
};
} // namespace nano
