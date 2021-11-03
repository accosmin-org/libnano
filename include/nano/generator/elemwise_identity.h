#pragma once

#include <nano/generator/elemwise.h>

namespace nano
{
    ///
    /// \brief forward the single-label original features as they are.
    ///
    class NANO_PUBLIC sclass_identity_t : public elemwise_input_sclass_t, public generated_sclass_t
    {
    public:

        feature_t feature(tensor_size_t ifeature) const override;

        auto process(tensor_size_t ifeature) const
        {
            const auto colsize = mapped_classes(ifeature);
            const auto process = [=] (const auto& label)
            {
                return static_cast<int32_t>(label);
            };

            return std::make_tuple(process, colsize);
        }
    };

    ///
    /// \brief forward the multi-label original features as they are.
    ///
    class NANO_PUBLIC mclass_identity_t : public elemwise_input_mclass_t, public generated_mclass_t
    {
    public:

        feature_t feature(tensor_size_t ifeature) const override;

        auto process(tensor_size_t ifeature) const
        {
            const auto colsize = mapped_classes(ifeature);
            const auto process = [=] (const auto& hits, auto&& storage)
            {
                this->copy(hits, storage);
            };

            return std::make_tuple(process, colsize);
        }

    private:

        template <typename thits, typename tstorage>
        static void copy(const thits& hits, tstorage& storage)
        {
            storage = hits.array().template cast<typename tstorage::Scalar>();
        }
    };

    ///
    /// \brief forward the scalar continuous original features as they are.
    ///
    class NANO_PUBLIC scalar_identity_t : public elemwise_input_scalar_t, public generated_scalar_t
    {
    public:

        feature_t feature(tensor_size_t ifeature) const override;

        auto process(tensor_size_t) const
        {
            const auto colsize = tensor_size_t{1};
            const auto process = [=] (const auto& values)
            {
                return static_cast<scalar_t>(values(0));
            };

            return std::make_tuple(process, colsize);
        }
    };

    ///
    /// \brief forward the structured continuous original features as they are.
    ///
    class NANO_PUBLIC struct_identity_t : public elemwise_input_struct_t, public generated_struct_t
    {
    public:

        feature_t feature(tensor_size_t ifeature) const override;

        auto process(tensor_size_t ifeature) const
        {
            const auto colsize = size(mapped_dims(ifeature));
            const auto process = [=] (const auto& values, auto&& storage)
            {
                storage = values.array().template cast<scalar_t>();
            };

            return std::make_tuple(process, colsize);
        }
    };
}
