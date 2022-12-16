#pragma once

#include <nano/learner.h>
#include <nano/model/cluster.h>

namespace nano
{
    class wlearner_t;
    using rwlearner_t = std::unique_ptr<wlearner_t>;

    ///
    /// \brief a weak learner is a machine learning model:
    ///     - parametrized by either a single feature or a small subset of features,
    ///     - easy to fit to the given residuals (aka the solution can be found analytically),
    ///     - with rather low accuracy that can be boosted by assemblying many of them using e.g. GradientBoosting.
    ///
    class NANO_PUBLIC wlearner_t : public learner_t, public clonable_t<wlearner_t>
    {
    public:
        ///
        /// \brief default constructor.
        ///
        explicit wlearner_t(string_t id);

        ///
        /// \brief returns the available implementations.
        ///
        static factory_t<wlearner_t>& all();

        ///
        /// \brief split the given samples using the currently selected features.
        ///
        /// NB: the given sample indices and the returned (cluster) splits
        ///     are relative to the whole dataset in the range [0, dataset.samples()).
        ///
        cluster_t split(const dataset_t&, const indices_t&) const;

        ///
        /// \brief compute the predictions for the given samples and add them to the given output buffer.
        ///
        /// NB: the given sample indices
        ///     are relative to the whole dataset in the range [0, dataset.samples()).
        ///
        tensor4d_t predict(const dataset_t&, const indices_cmap_t&) const;
        void       predict(const dataset_t&, const indices_cmap_t&, tensor4d_map_t) const;

        ///
        /// \brief select the feature or the features and estimate their associated parameters
        ///     that matches the best the given residuals/gradients in terms of the L2-norm
        ///     using the given sample indices:
        ///
        ///     argmin_h mean(L2-norm(-gradients(i), h(inputs(i))), i in indices)
        ///
        ///     where h is the weak learner.
        ///
        /// NB: returns how well the fitted weak learner matches the residuals - the lower, the better.
        ///
        /// NB: the given sample indices and gradients
        ///     are relative to the whole dataset in the range [0, dataset.samples()).
        ///
        scalar_t fit(const dataset_t&, const indices_t&, const tensor4d_t& gradients);

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
        /// \brief score that indicates fitting failed (e.g. unsupported feature types).
        ///
        static constexpr scalar_t no_fit_score() { return std::numeric_limits<scalar_t>::max(); }

    protected:
        virtual scalar_t  do_fit(const dataset_t&, const indices_t&, const tensor4d_t&)             = 0;
        virtual cluster_t do_split(const dataset_t&, const indices_t&) const                        = 0;
        virtual void      do_predict(const dataset_t&, const indices_cmap_t&, tensor4d_map_t) const = 0;
    };
} // namespace nano
