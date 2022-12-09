#pragma once

#include <nano/wlearner/hash.h>
#include <nano/wlearner/single.h>

namespace nano
{
    ///
    /// \brief a (look-up) table is a weak learner that returns a constant per labeling:
    ///     table(x) =
    ///     {
    ///         tables[hash(x[feature])], if x[feature] is given and its hash is within a (sub)set of hashes,
    ///         zero, otherwise (if the feature is missing)
    ///     }
    ///
    ///     where feature is the selected discrete feature.
    ///
    /// NB: both single-label and multi-label discrete features are supported.
    /// NB: continuous features and missing feature values are skipped during fiting.
    ///
    class NANO_PUBLIC table_wlearner_t : public single_feature_wlearner_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        explicit table_wlearner_t(string_t id);

        ///
        /// \brief @see estimator_t
        ///
        std::istream& read(std::istream&) override;

        ///
        /// \brief @see estimator_t
        ///
        std::ostream& write(std::ostream&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        cluster_t split(const dataset_t&, const indices_t&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        void predict(const dataset_t&, const indices_cmap_t&, tensor4d_map_t) const override;

        ///
        /// \brief returns the hashes of the distinct single-class or multi-class labeling.
        ///
        const wlearner::hashes_t& hashes() const { return m_hashes; }

        ///
        /// \brief returns the mapping of label hashes to tables of coefficients.
        ///
        const indices_t& hash2tables() const { return m_hash2tables; }

    protected:
        class cache_t;

        scalar_t set(const dataset_t&, const indices_t&, const cache_t&);

    private:
        // attributes
        wlearner::hashes_t m_hashes;      ///< hashes of the distinct multi-class labeling (during fitting)
        indices_t          m_hash2tables; ///< map label hashes to tables of coefficients
    };

    ///
    /// \brief the dense (look-up) table weak learner fits a constant
    ///     for each posible labeling, thus the set of hashes is all found in the training samples.
    ///
    class NANO_PUBLIC dense_table_wlearner_t final : public table_wlearner_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        dense_table_wlearner_t();

        ///
        /// \brief @see clonable_t
        ///
        rwlearner_t clone() const override;

        ///
        /// \brief @see wlearner_t
        ///
        scalar_t fit(const dataset_t&, const indices_t&, const tensor4d_t&) override;
    };

    ///
    /// \brief the k-best (look-up) table weak learner fits a constant
    ///     only to the best labeling subset.
    ///
    /// NB: the number of best labeling to consider should be tuned (typically proportional to its capacity).
    ///
    class NANO_PUBLIC kbest_table_wlearner_t final : public table_wlearner_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        kbest_table_wlearner_t();

        ///
        /// \brief @see clonable_t
        ///
        rwlearner_t clone() const override;

        ///
        /// \brief @see wlearner_t
        ///
        scalar_t fit(const dataset_t&, const indices_t&, const tensor4d_t&) override;
    };

    ///
    /// \brief the k-split (look-up) table weak learner fits a constant
    ///     to groups of labeling that are the most coherent given the training samples.
    ///
    /// NB: the number of splits (groups) should be tuned (typically proportional to its capacity).
    ///
    class NANO_PUBLIC ksplit_table_wlearner_t final : public table_wlearner_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        ksplit_table_wlearner_t();

        ///
        /// \brief @see clonable_t
        ///
        rwlearner_t clone() const override;

        ///
        /// \brief @see wlearner_t
        ///
        scalar_t fit(const dataset_t&, const indices_t&, const tensor4d_t&) override;
    };

    ///
    /// \brief the discrete step (look-up) weak learner fits a constant
    ///     only to the best labeling.
    ///
    /// NB: equivalent to k-best version with k set to 1.
    /// NB: this weak learner is inspired by the MARS algorithm extend to handle discrete/categorical features:
    ///     see "Multivariate adaptive regression splines", by Jerome Friedman
    ///
    class NANO_PUBLIC dstep_table_wlearner_t final : public table_wlearner_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        dstep_table_wlearner_t();

        ///
        /// \brief @see clonable_t
        ///
        rwlearner_t clone() const override;

        ///
        /// \brief @see wlearner_t
        ///
        scalar_t fit(const dataset_t&, const indices_t&, const tensor4d_t&) override;
    };
} // namespace nano
