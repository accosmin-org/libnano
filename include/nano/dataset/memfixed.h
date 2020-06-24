#pragma once

#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief in-memory dataset consisting of fixed-size inputs with optional targets
    ///     split into training, validation and testing parts.
    ///
    /// NB: the internal storage type can be different than scalar_t,
    ///     for example the most efficient way of storing RGB or grayscale image datasets
    ///     is to use one byte per color channel and pixel.
    ///
    /// NB: the customization point (in the derived classes) consists
    ///     of generating/loading the inputs and the targets and
    ///     of generating the training, validation and test dataset splits.
    ///
    template <typename tscalar>
    class memfixed_dataset_t : public dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        memfixed_dataset_t() = default;

        ///
        /// \brief load dataset in memory.
        ///
        virtual bool load() = 0;

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] feature_t ifeature(const tensor_size_t index) const override
        {
            const auto idim = this->idim();
            assert(index >= 0 && ::nano::size(idim));

            const auto modx = std::get<1>(idim) * std::get<2>(idim);
            const auto mody = std::get<2>(idim);
            const auto x = index / modx;
            const auto y = (index - x * modx) / mody;
            const auto z = index - x * modx - y * mody;

            feature_t feature{scat("feature_", x, "_", y, "_", z)};
            return feature;
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor_size_t samples() const override
        {
            return m_inputs.template size<0>();
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor_size_t samples(fold_t fold) const override
        {
            return indices(fold).size();
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor4d_t inputs(fold_t fold) const override
        {
            return m_inputs.template indexed<scalar_t>(indices(fold));
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor4d_t inputs(fold_t fold, tensor_range_t range) const override
        {
            const auto segment = indices(fold).slice(range);
            return m_inputs.template indexed<scalar_t>(segment);
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor1d_t inputs(fold_t fold, tensor_range_t range, tensor_size_t feature) const override
        {
            assert(feature >= 0 && feature < features());

            const auto imatrix = m_inputs.reshape(samples(), features()).matrix();
            const auto segment = indices(fold).slice(range);

            tensor1d_t fvalues(segment.size());
            for (tensor_size_t i = 0, size = segment.size(); i < size; ++ i)
            {
                fvalues(i) = imatrix(segment(i), feature);
            }

            return fvalues;
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor2d_t inputs(fold_t fold, tensor_range_t range, const indices_t& features) const override
        {
            assert(features.min() >= 0 && features.max() < this->features());

            const auto imatrix = m_inputs.reshape(samples(), this->features()).matrix();
            const auto segment = indices(fold).slice(range);

            tensor2d_t fvalues(segment.size(), features.size());
            for (tensor_size_t i = 0, size = segment.size(); i < size; ++ i)
            {
                for (tensor_size_t f = 0; f < features.size(); ++ f)
                {
                    fvalues(i, f) = imatrix(segment(i), features(f));
                }
            }

            return fvalues;
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor4d_t targets(fold_t fold) const override
        {
            return m_targets.indexed<scalar_t>(indices(fold));
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor4d_t targets(fold_t fold, tensor_range_t range) const override
        {
            const auto segment = indices(fold).slice(range);
            return m_targets.indexed<scalar_t>(segment);
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor3d_dim_t idim() const override
        {
            return make_dims(m_inputs.template size<1>(), m_inputs.template size<2>(), m_inputs.template size<3>());
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor3d_dim_t tdim() const override
        {
            return make_dims(m_targets.size<1>(), m_targets.size<2>(), m_targets.size<3>());
        }

        ///
        /// \brief returns the inputs and targets as they are stored.
        ///
        [[nodiscard]] const auto& all_inputs() const { return m_inputs; }
        [[nodiscard]] const auto& all_targets() const { return m_targets; }

        ///
        /// \brief returns the mutable input and target sample.
        ///
        auto input(tensor_size_t sample) { return m_inputs.tensor(sample); }
        auto target(tensor_size_t sample) { return m_targets.tensor(sample); }

        ///
        /// \brief returns the constant input and target sample.
        ///
        [[nodiscard]] auto input(tensor_size_t sample) const { return m_inputs.tensor(sample); }
        [[nodiscard]] auto target(tensor_size_t sample) const { return m_targets.tensor(sample); }

    protected:

        ///
        /// \brief allocate input and target tensors
        ///
        void resize(const tensor4d_dim_t& idim, const tensor4d_dim_t& tdim)
        {
            assert(std::get<0>(idim) == std::get<0>(tdim));

            m_inputs.resize(idim);
            m_targets.resize(tdim);
        }

    private:

        // attributes
        tensor_mem_t<tscalar, 4>    m_inputs;       ///< (total number of samples, #idim1, #idim2, #idim3)
        tensor_mem_t<scalar_t, 4>   m_targets;      ///< (total number of samples, #tdim1, #tdim2, #tdim3)
    };
}
