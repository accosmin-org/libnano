#pragma once

#include <nano/tpool.h>
#include <nano/parameter.h>
#include <nano/mlearn/fold.h>
#include <nano/mlearn/split.h>
#include <nano/mlearn/feature.h>
#include <nano/mlearn/elemwise.h>

namespace nano
{
    ///
    /// \brief machine learning dataset consisting of a collection of samples
    ///     split into training, validation and testing parts.
    ///
    /// NB: the samples are organized by folds, specified by:
    ///     - the index and
    ///     - the protocol (training, validation or test).
    ///
    /// NB: each sample consists of:
    ///     - a fixed number of (input) feature values and
    ///     - optionally a target if a supervised ML task.
    ///
    class dataset_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        dataset_t() = default;

        ///
        /// \brief disable copying
        ///
        dataset_t(const dataset_t&) = default;
        dataset_t& operator=(const dataset_t&) = default;

        ///
        /// \brief enable moving
        ///
        dataset_t(dataset_t&&) noexcept = default;
        dataset_t& operator=(dataset_t&&) noexcept = default;

        ///
        /// \brief default destructor
        ///
        virtual ~dataset_t() = default;

        ///
        /// \brief returns the number of folds.
        ///
        [[nodiscard]] virtual size_t folds() const
        {
            return m_splits.size();
        }

        ///
        /// \brief set the number of folds.
        ///
        void folds(const size_t folds)
        {
            m_folds = folds;
            m_splits = std::vector<split_t>(folds, split_t{});
        }

        ///
        /// \brief randomly shuffle the samples associated to a given fold.
        ///
        void shuffle(fold_t fold)
        {
            auto& indices = this->indices(fold);
            std::shuffle(begin(indices), end(indices), make_rng());
        }

        ///
        /// \brief returns the percentage of training samples.
        ///
        [[nodiscard]] tensor_size_t train_percentage() const
        {
            return m_train.get();
        }

        ///
        /// \brief set the percentage of training samples.
        ///
        void train_percentage(const tensor_size_t train_percentage)
        {
            m_train = train_percentage;
        }

        ///
        /// \brief returns the total number of samples.
        ///
        [[nodiscard]] virtual tensor_size_t samples() const = 0;

        ///
        /// \brief returns the total number of samples of the given fold.
        ///
        [[nodiscard]] virtual tensor_size_t samples(fold_t) const = 0;

        ///
        /// \brief returns the dimension of a sample.
        ///
        [[nodiscard]] virtual tensor3d_dim_t idim() const = 0;

        ///
        /// \brief returns the dimension of the target (if provided).
        ///
        [[nodiscard]] virtual tensor3d_dim_t tdim() const = 0;

        ///
        /// \brief returns the feature description of the target (if provided).
        ///
        [[nodiscard]] virtual feature_t tfeature() const = 0;

        ///
        /// \brief returns the toal number of input features.
        ///
        [[nodiscard]] tensor_size_t features() const { return ::nano::size(idim()); }

        ///
        /// \brief returns the feature description of a given feature index.
        ///
        /// NB: the feature index must be in the range [0, features()).
        ///
        [[nodiscard]] virtual feature_t ifeature(tensor_size_t index) const = 0;

        ///
        /// \brief returns the feature values of all samples of a given fold.
        ///
        [[nodiscard]] virtual tensor4d_t inputs(fold_t) const = 0;

        ///
        /// \brief returns the feature values of the given range of samples.
        ///
        /// NB: the range of samples must be included in [0, samples(fold)).
        ///
        [[nodiscard]] virtual tensor4d_t inputs(fold_t, tensor_range_t) const = 0;

        ///
        /// \brief returns the feature values of the given range of samples.
        ///
        /// NB: the range of samples must be included in [0, samples(fold)).
        ///
        [[nodiscard]] virtual tensor1d_t inputs(fold_t, tensor_range_t, tensor_size_t feature) const = 0;

        ///
        /// \brief returns the feature values of the given range of samples.
        ///
        /// NB: the range of samples must be included in [0, samples(fold)).
        ///
        [[nodiscard]] virtual tensor2d_t inputs(fold_t, tensor_range_t, const indices_t& features) const = 0;

        ///
        /// \brief returns the targets of all samples of a given fold.
        ///
        [[nodiscard]] virtual tensor4d_t targets(fold_t) const = 0;

        ///
        /// \brief returns the targets of the given range of samples.
        ///
        /// NB: the range must be included in [0, samples(fold)).
        ///
        [[nodiscard]] virtual tensor4d_t targets(fold_t, tensor_range_t) const = 0;

        ///
        /// \brief returns the element-wise statistics for all inputs of the given fold.
        ///
        /// NB: e.g. this is useful for normalizing the inputs to zero mean and unit variance.
        ///
        [[nodiscard]] elemwise_stats_t istats(fold_t fold, tensor_size_t batch) const
        {
            std::vector<elemwise_stats_t> stats(tpool_t::size(), elemwise_stats_t{idim()});
            loop(execution::par, fold, batch, [&] (tensor_range_t range, size_t tnum)
            {
                stats[tnum].update(inputs(fold, range));
            });

            std::for_each(++ stats.begin(), stats.end(), [&] (const elemwise_stats_t& tstats)
            {
                stats[0].update(tstats);
            });
            return stats[0].done(samples(fold));
        }

        ///
        /// \brief iterate through all the samples of the given fold by calling
        ///     the given operator for each batch of samples of size `batch` (except maybe the last one).
        ///
        template <typename toperator>
        void loop(const execution policy, fold_t fold, tensor_size_t batch, const toperator& op) const
        {
            assert(batch > 0);

            const auto samples = this->samples(fold);

            switch (policy)
            {
            case execution::par:
                loopr(samples, batch, [&] (tensor_size_t begin, tensor_size_t end, size_t tnum)
                {
                    op(::nano::make_range(begin, end), static_cast<size_t>(tnum));
                });
                break;

            default:
                for (tensor_size_t begin = 0, end = 0; begin < samples; begin = end)
                {
                    end = std::min(begin + batch, samples);
                    op(::nano::make_range(begin, end), 0U);
                }
                break;
            }
        }

    protected:

        split_t& split(size_t fold) { return m_splits.at(fold); }
        [[nodiscard]] const split_t& split(size_t fold) const { return m_splits.at(fold); }

        indices_t& indices(fold_t fold) { return split(fold.m_index).indices(fold); }
        [[nodiscard]] const indices_t& indices(fold_t fold) const { return split(fold.m_index).indices(fold); }

    private:

        // attributes
        uparam1_t   m_folds{"dataset::folds", 1, LE, 10, LE, 100};  ///< number of folds
        iparam1_t   m_train{"dataset::trper", 10, LE, 80, LE, 90};  ///< percentage of training samples (excluding the test samples)
        splits_t    m_splits{10};                                   ///<
    };
}
