#pragma once

#include <nano/feature.h>
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
        /// \brief enable copying
        ///
        memfixed_dataset_t() = default;
        memfixed_dataset_t(const memfixed_dataset_t&) = default;
        memfixed_dataset_t& operator=(const memfixed_dataset_t&) = default;

        ///
        /// \brief enable moving
        ///
        memfixed_dataset_t(memfixed_dataset_t&&) noexcept = default;
        memfixed_dataset_t& operator=(memfixed_dataset_t&&) noexcept = default;

        ///
        /// \brief default destructor
        ///
        virtual ~memfixed_dataset_t() = default;

        ///
        /// \brief load dataset in memory
        ///
        virtual bool load() = 0;

        ///
        /// \brief returns the description of the target feature (if a supervised task)
        ///
        virtual feature_t tfeature() const = 0;

        ///
        /// \brief returns the total number of samples
        ///
        tensor_size_t samples() const
        {
            return m_inputs.template size<0>();
        }

        ///
        /// \brief returns the number of samples associated to a given fold
        ///
        tensor_size_t samples(const fold_t& fold) const
        {
            return indices(fold).size();
        }

        ///
        /// \brief returns the inputs tensor for all samples in the given fold
        ///
        tensor4d_t inputs(const fold_t& fold) const
        {
            return m_inputs.template indexed<scalar_t>(indices(fold));
        }

        ///
        /// \brief returns the inputs tensor for the [begin, end) range of samples in the given fold
        ///
        tensor4d_t inputs(const fold_t& fold, const tensor_size_t begin, const tensor_size_t end) const
        {
            return m_inputs.template indexed<scalar_t>(indices(fold).segment(begin, end - begin));
        }

        ///
        /// \brief returns the targets tensor for all samples in the given fold (if a supervised task)
        ///
        tensor4d_t targets(const fold_t& fold) const
        {
            return m_targets.indexed<scalar_t>(indices(fold));
        }

        ///
        /// \brief returns the targets tensor for the [begin, end) range of samples in the given fold (if a supervised task)
        ///
        tensor4d_t targets(const fold_t& fold, const tensor_size_t begin, const tensor_size_t end) const
        {
            return m_targets.indexed<scalar_t>(indices(fold).segment(begin, end - begin));
        }

        ///
        /// \brief returns the 3D dimension of a sample's input tensor
        ///
        auto idim() const
        {
            return make_dims(m_inputs.template size<1>(), m_inputs.template size<2>(), m_inputs.template size<3>());
        }

        ///
        /// \brief returns the 3D dimension of a sample's target tensor (if a supervised task)
        ///
        auto tdim() const
        {
            return make_dims(m_targets.size<1>(), m_targets.size<2>(), m_targets.size<3>());
        }

        ///
        /// \brief returns the inputs and the targets as they are stored
        ///
        const auto& inputs() const { return m_inputs; }
        const auto& targets() const { return m_targets; }

        auto input(const tensor_size_t sample) { return m_inputs.tensor(sample); }
        auto target(const tensor_size_t sample) { return m_targets.tensor(sample); }

        auto input(const tensor_size_t sample) const { return m_inputs.tensor(sample); }
        auto target(const tensor_size_t sample) const { return m_targets.tensor(sample); }

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
