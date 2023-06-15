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
    ///
    /// \brief default constructor (use all available features).
    ///
    explicit pairwise_product_t();

    ///
    /// \brief constructor (use pairs of the given features, if of the appropriate type).
    ///
    pairwise_product_t(indices_t original_features);

    ///
    /// \brief constructor (use the given pairs of features, if of the appropriate type).
    ///
    pairwise_product_t(indices_t original_features1, indices_t original_features2);

    ///
    /// \brief @see generator_t
    ///
    feature_t feature(tensor_size_t ifeature) const override;

    static auto process(tensor_size_t)
    {
        const auto colsize = tensor_size_t{1};
        const auto process = [](const auto& values1, const auto& values2)
        { return static_cast<scalar_t>(values1(0)) * static_cast<scalar_t>(values2(0)); };

        return std::make_tuple(process, colsize);
    }
};

using pairwise_product_generator_t = pairwise_generator_t<pairwise_product_t>;
} // namespace nano
