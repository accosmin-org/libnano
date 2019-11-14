#pragma once

#include <nano/mlearn.h>

namespace nano
{
    ///
    /// \brief interface to iterate through a collection of samples associated to a machine learning dataset.
    ///
    /// NB: the collection of samples is usually a fold.
    /// NB: a sample is specified by a set of input features and optionally by a target.
    ///
    class iterator_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        iterator_t() = default;

        ///
        /// \brief disable copying
        ///
        iterator_t(const iterator_t&) = delete;
        iterator_t& operator=(const iterator_t&) = delete;

        ///
        /// \brief enable moving
        ///
        iterator_t(iterator_t&&) noexcept = default;
        iterator_t& operator=(iterator_t&&) noexcept = default;

        ///
        /// \brief default destructor
        ///
        virtual ~iterator_t() = default;

        ///
        /// \brief returns the total number of folds
        ///
        virtual size_t folds() const = 0;

        ///
        /// \brief shuffle the samples of the given fold
        ///
        virtual void shuffle(const fold_t&) const = 0;

        ///
        /// \brief returns the total number of samples of the given fold
        ///
        virtual tensor_size_t samples(const fold_t&) const = 0;

        ///
        /// \brief returns the input dimension of a sample
        ///
        virtual tensor3d_dim_t idim() const = 0;

        ///
        /// \brief returns the target dimension of a sample
        ///
        virtual tensor3d_dim_t tdim() const = 0;

        ///
        /// \brief returns the inputs (or the input features) for the [begin, end) range of samples of the given fold
        ///
        virtual void inputs(const fold_t&, tensor_size_t begin, tensor_size_t end, tensor4d_t&) const = 0;

        ///
        /// \brief returns the targets for the [begin, end) range of samples of the given fold
        ///
        virtual void targets(const fold_t&, tensor_size_t begin, tensor_size_t end, tensor4d_t&) const = 0;
    };
}
