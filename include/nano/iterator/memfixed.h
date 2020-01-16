#pragma once

#include <nano/iterator.h>
#include <nano/memfixed.h>

namespace nano
{
    ///
    /// \brief sample iterator using a dataset with fixed-sized inputs without any pre-processing.
    ///
    template <typename tscalar>
    class memfixed_iterator_t final : public iterator_t
    {
    public:

        using source_t = memfixed_dataset_t<tscalar>;

        ///
        /// \brief constructor
        ///
        explicit memfixed_iterator_t(source_t& source) :
            m_source(source)
        {
        }

        ///
        /// \brief @see iterator_t
        ///
        size_t folds() const override
        {
            return m_source.folds();
        }

        ///
        /// \brief @see iterator_t
        ///
        void shuffle(const fold_t& fold) const override
        {
            m_source.shuffle(fold);
        }

        ///
        /// \brief @see iterator_t
        ///
        tensor_size_t samples(const fold_t& fold) const override
        {
            return m_source.samples(fold);
        }

        ///
        /// \brief @see iterator_t
        ///
        tensor3d_dim_t idim() const override
        {
            return m_source.idim();
        }

        ///
        /// \brief @see iterator_t
        ///
        tensor3d_dim_t tdim() const override
        {
            return m_source.tdim();
        }

        ///
        /// \brief @see iterator_t
        ///
        tensor4d_t inputs(const fold_t& fold, tensor_size_t begin, tensor_size_t end) const override
        {
            return m_source.inputs(fold, begin, end);
        }

        ///
        /// \brief @see iterator_t
        ///
        tensor4d_t targets(const fold_t& fold, tensor_size_t begin, tensor_size_t end) const override
        {
            return m_source.targets(fold, begin, end);
        }

    private:

        // attributes
        source_t&           m_source;   ///< source dataset
    };
}
