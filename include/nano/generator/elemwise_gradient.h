#pragma once

#include <nano/generator/elemwise.h>
#include <nano/generator/gradient.h>

namespace nano
{
    ///
    /// \brief generate image gradient-like structured features:
    ///     - vertical and horizontal gradients,
    ///     - edge orientation and magnitude.
    ///
    class NANO_PUBLIC elemwise_gradient_t : public elemwise_input_struct_t, public generated_struct_t
    {
    public:

        explicit elemwise_gradient_t(
            kernel3x3_type = kernel3x3_type::sobel,
            indices_t original_features = indices_t{});

        feature_t feature(tensor_size_t ifeature) const override;

        auto process(tensor_size_t ifeature) const
        {
            const auto dims = mapped_dims(ifeature);
            const auto mode = mapped_mode(ifeature);
            const auto channel = mapped_channel(ifeature);
            const auto kernel = make_kernel3x3<scalar_t>(m_type);

            const auto rows = std::get<1>(dims);
            const auto cols = std::get<2>(dims);
            const auto colsize = rows * cols;
            const auto process = [=] (const auto& values, auto&& storage)
            {
                gradient3x3(mode, values.tensor(channel), kernel, map_tensor(storage.data(), rows, cols));
            };

            return std::make_tuple(process, colsize);
        }

    private:

        feature_mapping_t do_fit() override;

        tensor_size_t mapped_channel(tensor_size_t ifeature) const;
        gradient3x3_mode mapped_mode(tensor_size_t ifeature) const;

        // attributes
        kernel3x3_type  m_type{kernel3x3_type::sobel};      ///<
        indices_t       m_original_features;                ///<
    };
}
