#pragma once

#include <nano/mlearn.h>
#include <nano/parameter.h>

namespace nano
{
    ///
    /// \brief machine learning dataset consisting of a collection of samples
    ///     split into training, validation and testing parts.
    ///
    class dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        dataset_t() = default;

        ///
        /// \brief returns the number of folds
        ///
        size_t folds() const
        {
            return m_splits.size();
        }

        ///
        /// \brief set the number of folds
        ///
        void folds(const size_t folds)
        {
            m_folds = folds;
            m_splits = std::vector<split_t>(folds, split_t{});
        }

        ///
        /// \brief randomly shuffle the samples associated to a given fold
        ///
        void shuffle(const fold_t& fold)
        {
            auto& indices = this->indices(fold);
            std::shuffle(begin(indices), end(indices), make_rng());
        }

        ///
        /// \brief returns the percentage of training samples
        ///
        tensor_size_t train_percentage() const
        {
            return m_trper.get();
        }

        ///
        /// \brief set the percentage of training samples
        ///
        void train_percentage(const tensor_size_t train_percentage)
        {
            m_trper = train_percentage;
        }

    protected:

        split_t& split(const size_t fold) { return m_splits.at(fold); }
        const split_t& split(const size_t fold) const { return m_splits.at(fold); }

        indices_t& indices(const fold_t& fold) { return split(fold.m_index).indices(fold); }
        const indices_t& indices(const fold_t& fold) const { return split(fold.m_index).indices(fold); }

    private:

        // attributes
        uparam1_t   m_folds{"dataset::folds", 1, LE, 10, LE, 100};      ///< number of folds
        iparam1_t   m_trper{"dataset::train_per", 10, LE, 80, LE, 90};  ///< percentage of training samples (excluding the test samples)
        splits_t    m_splits{10};                                       ///<
    };
}
