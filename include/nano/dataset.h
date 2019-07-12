#pragma once

#include <nano/arch.h>
#include <nano/json.h>
#include <nano/mlearn.h>
#include <nano/factory.h>
#include <nano/feature.h>

namespace nano
{
    class dataset_t;
    using dataset_factory_t = factory_t<dataset_t>;
    using rdataset_t = dataset_factory_t::trobject;

    ///
    /// \brief machine learning dataset consisting of a collection of fixed-size 3D input tensors
    ///     split into training, validation and testing parts.
    ///
    /// dataset_t's interface handles the following cases:
    ///     - both supervised and unsupervised machine learning tasks
    ///     - categorical and continuous features
    ///     - missing feature values
    ///
    class NANO_PUBLIC dataset_t : public json_configurable_t
    {
    public:

        ///
        /// \brief returns the available implementations
        ///
        static dataset_factory_t& all();

        ///
        /// \brief populate the dataset with samples
        ///
        virtual bool load() = 0;

        ///
        /// \brief returns the number of folds
        ///
        virtual size_t folds() const = 0;

        ///
        /// \brief returns the total number of input features
        ///
        virtual size_t ifeatures() const = 0;

        ///
        /// \brief returns the description of the given input feature
        ///
        virtual feature_t ifeature(const size_t index) const = 0;

        ///
        /// \brief returns the description of the target feature
        ///
        virtual feature_t tfeature() const = 0;

        ///
        /// \brief returns the inputs tensor of a given fold
        ///
        virtual tensor4d_t inputs(const fold_t&) const = 0;

        ///
        /// \brief returns the targets tensor of a given fold (if a supervised task)
        ///
        virtual tensor4d_t targets(const fold_t&) const = 0;

        ///
        /// \brief randomly shuffle the samples associated for the given fold
        ///
        virtual void shuffle(const fold_t&) = 0;
    };
}
