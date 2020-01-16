#pragma once

#include <nano/tpool.h>
#include <nano/mlearn/fold.h>
#include <nano/mlearn/elemwise.h>

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
        virtual tensor4d_t inputs(const fold_t&, tensor_size_t begin, tensor_size_t end) const = 0;

        ///
        /// \brief returns the targets for the [begin, end) range of samples of the given fold
        ///
        virtual tensor4d_t targets(const fold_t&, tensor_size_t begin, tensor_size_t end) const = 0;

        ///
        /// \brief returns the element-wise statistics for all inputs of the given fold.
        ///
        /// NB: this is useful for normalizing the inputs to zero mean and unit variance.
        ///
        elemwise_stats_t istats(const fold_t& fold, const tensor_size_t batch) const
        {
            std::vector<elemwise_stats_t> stats(tpool_t::size(), elemwise_stats_t{idim()});

            loop(fold, batch, [&] (const auto& inputs, const auto&, const auto, const auto, const auto tnum)
            {
                stats[tnum].update(inputs);
            }, execution::par);

            std::for_each(++ stats.begin(), stats.end(), [&] (const auto& tstats)
            {
                stats[0].update(tstats);
            });

            stats[0].done(samples(fold));
            return stats[0];
        }

        ///
        /// \brief iterate through the [begin, end) range of samples of a fold using multiple threads and
        ///     call the given operator like (inputs, targets, tbegin, tend, tnum)
        ///     where the [tbegin, tend) chunk of samples is of size `batch` (except maybe for the last one).
        ///
        template <typename toperator>
        void loop(const fold_t& fold, const tensor_size_t begin, const tensor_size_t end,
            const tensor_size_t batch, const toperator& op, const execution policy) const
        {
            assert(begin >= 0 && begin < end && end <= samples(fold));

            switch (policy)
            {
            case execution::par:
                loopr(end - begin, batch, [&] (tensor_size_t tbegin, tensor_size_t tend, tensor_size_t tnum)
                {
                    op( inputs(fold, begin + tbegin, begin + tend),
                        targets(fold, begin + tbegin, begin + tend),
                        begin + tbegin, begin + tend, static_cast<size_t>(tnum));
                });
                break;

            default:
                for (auto tbegin = begin, tend = begin; tbegin < end; tbegin = tend)
                {
                    tend = std::min(tbegin + batch, end);
                    op( inputs(fold, tbegin, tend),
                        targets(fold, tbegin, tend),
                        tbegin, tend, 0U);
                }
                break;
            }
        }

        ///
        /// \brief iterate through all samples of a fold using multiple threads and
        ///     call the given operator like (inputs, targets, tbegin, tend, tnum)
        ///     where the [tbegin, tend) chunk of samples is of size `batch` (except maybe for the last one).
        ///
        template <typename toperator>
        void loop(const fold_t& fold, const tensor_size_t batch, const toperator& op, const execution policy) const
        {
            loop(fold, 0, samples(fold), batch, op, policy);
        }
    };
}
