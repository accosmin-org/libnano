#pragma once

#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief wrapper over a dataset to shuffle (across samples) a given feature.
    ///
    /// NB: this is useful for estimating the importance of a feature by measuring
    ///     the difference in accuracy when the associated feature values are shuffled (across samples).
    ///
    class shuffle_dataset_t : public dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        shuffle_dataset_t(const dataset_t& source, tensor_size_t feature2shuffle) :
            m_source(source),
            m_feature2shuffle(feature2shuffle)
        {
            assert(feature2shuffle >= 0 && feature2shuffle < m_source.features());
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] size_t folds() const override
        {
            return m_source.folds();
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] feature_t ifeature(const tensor_size_t index) const override
        {
            return m_source.ifeature(index);
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] feature_t tfeature() const override
        {
            return m_source.tfeature();
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor_size_t samples() const override
        {
            return m_source.samples();
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor_size_t samples(fold_t fold) const override
        {
            return m_source.samples(fold);
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor4d_t inputs(fold_t fold) const override
        {
            auto inputs = m_source.inputs(fold);
            shuffle(inputs, m_feature2shuffle);
            return inputs;
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor4d_t inputs(fold_t fold, tensor_range_t range) const override
        {
            auto inputs = m_source.inputs(fold, range);
            shuffle(inputs, m_feature2shuffle);
            return inputs;
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor1d_t inputs(fold_t fold, tensor_range_t range, tensor_size_t feature) const override
        {
            auto inputs = m_source.inputs(fold, range, feature);
            if (m_feature2shuffle == feature)
            {
                shuffle(inputs, 0);
            }
            return inputs;
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor2d_t inputs(fold_t fold, tensor_range_t range, const indices_t& features) const override
        {
            auto inputs =  m_source.inputs(fold, range, features);
            const auto* const it = std::find(features.begin(), features.end(), m_feature2shuffle);
            if (it != features.end())
            {
                shuffle(inputs, std::distance(features.begin(), it));
            }
            return inputs;
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor4d_t targets(fold_t fold) const override
        {
            return m_source.targets(fold);
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor4d_t targets(fold_t fold, tensor_range_t range) const override
        {
            return m_source.targets(fold, range);
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor3d_dim_t idim() const override
        {
            return m_source.idim();
        }

        ///
        /// \brief @see dataset_t
        ///
        [[nodiscard]] tensor3d_dim_t tdim() const override
        {
            return m_source.tdim();
        }

    private:

        template <typename ttensor>
        void shuffle(ttensor& inputs, tensor_size_t col) const
        {
            const auto matrix = inputs.reshape(inputs.template size<0>(), -1);
            assert(col >= 0 && col < matrix.cols());

            auto&& rng = ::nano::make_rng();
            using diff_t = tensor_size_t;
            using dist_t = std::uniform_int_distribution<diff_t>;
            using param_t = dist_t::param_type;

            dist_t D;
            for (diff_t rows = matrix.rows(), row = rows - 1; row > 0; -- row)
            {
                using std::swap;
                swap(matrix(row, col), matrix(D(rng, param_t(0, row)), col));
            }
        }

        // attributes
        const dataset_t&    m_source;               ///< original dataset
        tensor_size_t       m_feature2shuffle{0};   ///< feature index to shuffle (across samples)
    };
}
