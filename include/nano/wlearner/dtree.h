#pragma once

#include <nano/wlearner.h>

namespace nano
{
    ///
    /// \brief node in the decision tree.
    ///
    struct dtree_node_t
    {
        tensor_size_t m_feature{-1};  ///< feature to evaluate (if a decision node)
        scalar_t      m_threshold{0}; ///< feature value threshold
        size_t        m_next{0};      ///< offset to the next node
        tensor_size_t m_table{-1};    ///< index in the prediction tables (if a leave)
    };

    using dtree_nodes_t = std::vector<dtree_node_t>;

    inline bool operator==(const dtree_node_t& lhs, const dtree_node_t& rhs)
    {
        return lhs.m_feature == rhs.m_feature && std::fabs(lhs.m_threshold - rhs.m_threshold) < 1e-8 &&
               lhs.m_next == rhs.m_next && lhs.m_table == rhs.m_table;
    }

    NANO_PUBLIC std::ostream& operator<<(std::ostream&, const dtree_node_t&);
    NANO_PUBLIC std::ostream& operator<<(std::ostream&, const dtree_nodes_t&);

    NANO_PUBLIC std::istream& read(std::istream&, dtree_node_t&);
    NANO_PUBLIC std::ostream& write(std::ostream&, const dtree_node_t&);

    ///
    /// \brief a decision tree is a weak learner that partitions the data using
    ///     decision stumps (for continuous scalar features only).
    ///
    /// NB: structured and discrete features and missing feature values are skipped during fiting.
    ///
    class NANO_PUBLIC dtree_wlearner_t final : public wlearner_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        dtree_wlearner_t();

        ///
        /// \brief @see estimator_t
        ///
        std::istream& read(std::istream&) override;

        ///
        /// \brief @see estimator_t
        ///
        std::ostream& write(std::ostream&) const override;

        ///
        /// \brief @see clonable_t
        ///
        rwlearner_t clone() const override;

        ///
        /// \brief @see wlearner_t
        ///
        void scale(const vector_t&) override;

        ///
        /// \brief @see wlearner_t
        ///
        cluster_t do_split(const dataset_t&, const indices_t&) const override;

        ///
        /// \brief @see wlearner_t
        ///
        void do_predict(const dataset_t&, const indices_cmap_t&, tensor4d_map_t) const override;

        ///
        /// \brief @see wlearner_t
        ///
        scalar_t do_fit(const dataset_t&, const indices_t&, const tensor4d_t&) override;

        ///
        /// \brief @see wlearner_t
        ///
        indices_t features() const override;

        ///
        /// \brief returns the flatten list of nodes (splitting and terminal ones).
        ///
        const dtree_nodes_t& nodes() const { return m_nodes; }

        ///
        /// \brief returns the table of coefficients of the terminal nodes.
        ///
        const tensor4d_t& tables() const { return m_tables; }

    private:
        // attributes
        dtree_nodes_t m_nodes;    ///< nodes in the decision tree
        tensor4d_t    m_tables;   ///< (#feature values, #outputs) - predictions at the leaves
        indices_t     m_features; ///< unique set of the selected features
    };
} // namespace nano
