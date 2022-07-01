#pragma once

#include <nano/dataset.h>
#include <nano/factory.h>
#include <nano/mlearn/cluster.h>
#include <nano/parameter.h>
#include <nano/stream.h>

namespace nano
{
    class wlearner_t;
    using wlearner_factory_t = factory_t<wlearner_t>;
    using rwlearner_t        = wlearner_factory_t::trobject;
    using wlearners_t        = std::vector<rwlearner_t>;

    ///
    /// \brief a weak learner is a machine learning model:
    ///     - parametrized by either a single feature or a small subset of features,
    ///     - easy to fit to the given residuals (aka the solution can be found analytically),
    ///     - with rather low accuracy that can be boosted by assemblying many of them using e.g. GradientBoosting.
    ///
    class NANO_PUBLIC wlearner_t : public serializable_t
    {
    public:
        ///
        /// \brief returns the available implementations.
        ///
        static wlearner_factory_t& all();

        ///
        /// \brief default constructor
        ///
        wlearner_t() = default;

        ///
        /// \brief @see serializable_t
        ///
        void read(std::istream&) override;

        ///
        /// \brief @see serializable_t
        ///
        void write(std::ostream&) const override;

        ///
        /// \brief clone the object.
        ///
        virtual rwlearner_t clone() const = 0;

        ///
        /// \brief split the given samples using the currently selected features.
        ///
        /// NB: the given sample indices and the returned (cluster) splits
        ///     are relative to the whole dataset in the range [0, dataset.samples()).
        ///
        virtual cluster_t split(const dataset_t&, const indices_t&) const = 0;

        ///
        /// \brief compute the predictions for the given samples and add them to the given output buffer.
        ///
        /// NB: the given sample indices
        ///     are relative to the whole dataset in the range [0, dataset.samples()).
        ///
        tensor4d_t   predict(const dataset_t&, const indices_cmap_t&) const;
        virtual void predict(const dataset_t&, const indices_cmap_t&, tensor4d_map_t) const = 0;

        ///
        /// \brief select the feature or the features and estimate their associated parameters
        ///     that matches the best the given residuals/gradients in terms of the L2-norm
        ///     using the given sample indices:
        ///
        ///     argmin_h mean(L2-norm(-gradients(i), h(inputs(i))), i in indices)
        ///
        ///     where h is the weak learner.
        ///
        /// NB: the given sample indices and gradients
        ///     are relative to the whole dataset in the range [0, dataset.samples()).
        ///
        virtual scalar_t fit(const dataset_t&, const indices_t&, const tensor4d_t& gradients) = 0;

        ///
        /// \brief adjust the weak learner's parameters to obtain linearly scaled predictions.
        ///
        /// NB: the scaling vector can be either:
        ///     - one dimensional, thus the same scale if applied to all splits or
        ///     - of the same dimension as the number of splits.
        ///
        virtual void scale(const vector_t&) = 0;

        ///
        /// \brief returns the selected features.
        ///
        virtual indices_t features() const = 0;

        ///
        /// \brief change the batch size (aka number of samples to process at a time).
        ///
        /// NB: this may require tuning for optimum speed.
        ///
        void batch(int batch);

        ///
        /// \brief score that indicates fitting failed (e.g. unsupported feature types).
        ///
        static constexpr scalar_t no_fit_score() { return std::numeric_limits<scalar_t>::max(); }

        ///
        /// \brief access functions
        ///
        auto batch() const { return m_batch.get(); }

    protected:
        static void check(const indices_t& samples);
        static void scale(tensor4d_t& tables, const vector_t& scale);

    private:
        // attributes
        iparam1_t m_batch{"wlearner::batch", 1, LE, 32, LE, 1024}; ///< batch size
    };
} // namespace nano
