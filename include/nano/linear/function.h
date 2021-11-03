#pragma once

#include <nano/loss.h>
#include <nano/function.h>
#include <nano/generator.h>
#include <nano/core/parameter.h>

namespace nano::linear
{
    ///
    /// \brief the ERM criterion used for optimizing the parameters of a linear model,
    ///     using a given loss function.
    ///
    /// NB: the ERM loss can be optionally regularized by penalizing:
    ///     - (1) the L1-norm of the weights matrix - like in LASSO
    ///     - (2) the L2-norm of the weights matrix - like in RIDGE (regression)
    ///     - (3) both the L1 and the L2-norms of the weights matrix - like in elastic net regularization
    ///     - (4) the variance of the loss values - like in VadaBoost
    ///
    class NANO_PUBLIC function_t final : public ::nano::function_t
    {
    public:

        ///
        /// \brief constructor
        ///
        function_t(const dataset_generator_t&, const loss_t&, flatten_cache_t&);

        ///
        /// \brief enable coying
        ///
        function_t(const function_t&) = default;
        function_t& operator=(const function_t&) = delete;

        ///
        /// \brief enable moving
        ///
        function_t(function_t&&) noexcept = default;
        function_t& operator=(function_t&&) noexcept = delete;

        ///
        /// \brief default destructor
        ///
        ~function_t() override = default;

        ///
        /// \brief extract the weight matrix from the given tensor
        ///
        template <typename ttensor>
        auto weights(ttensor& x) const
        {
            assert(x.size() == m_isize * m_tsize + m_tsize);
            return map_tensor(x.data(), m_isize, m_tsize);
        }

        template <typename ttensor>
        auto weights(const ttensor& x) const
        {
            assert(x.size() == m_isize * m_tsize + m_tsize);
            return map_tensor(x.data(), m_isize, m_tsize);
        }

        ///
        /// \brief extract the bias vector from the given tensor
        ///
        template <typename ttensor>
        auto bias(ttensor& x) const
        {
            assert(x.size() == m_isize * m_tsize + m_tsize);
            return map_tensor(x.data() + m_isize * m_tsize, m_tsize);
        }

        template <typename ttensor>
        auto bias(const ttensor& x) const
        {
            assert(x.size() == m_isize * m_tsize + m_tsize);
            return map_tensor(x.data() + m_isize * m_tsize, m_tsize);
        }

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

        ///
        /// \brief change parameters
        ///
        void l1reg(scalar_t l1reg) { m_l1reg.set(l1reg); }
        void l2reg(scalar_t l2reg) { m_l2reg.set(l2reg); }
        void vAreg(scalar_t vAreg) { m_vAreg.set(vAreg); }
        void scaling(scaling_type scaling) { m_scaling = scaling; }

        ///
        /// \brief access functions
        ///
        auto isize() const { return m_isize; }
        auto tsize() const { return m_tsize; }
        auto l1reg() const { return m_l1reg.get(); }
        auto l2reg() const { return m_l2reg.get(); }
        auto vAreg() const { return m_vAreg.get(); }
        auto scaling() const { return m_scaling; }
        const auto& loss() const { return m_loss; }
        const auto& stats() const { return m_stats; }
        const auto& generator() const { return m_generator; }

    private:

        using xdataset_t = dataset_generator_t;

        // attributes
        const xdataset_t&   m_dataset;          ///<
        const loss_t&       m_loss;             ///<
        flatten_cache_t&    m_flatten_cache;    ///< cache to buffer flatten inputs and targets
        flatten_stats_t     m_flatten_stats;    ///< element-wise statistics to be used for feature_scaling
        tensor_size_t       m_isize{0};         ///< #inputs (e.g. size of the flatten input feature tensor)
        tensor_size_t       m_tsize{0};         ///< #targets (e.g. size of the flatten target tensor, number of classes)
        sparam1_t           m_l1reg{"linear::L1", 0, LE, 0, LE, 1e+8};  ///< regularization factor - see (1), (3)
        sparam1_t           m_l2reg{"linear::L2", 0, LE, 0, LE, 1e+8};  ///< regularization factor - see (2), (3)
        sparam1_t           m_vAreg{"linear::VA", 0, LE, 0, LE, 1e+8};  ///< regularization factor - see (4)
        scaling_type        m_scaling{scaling_type::none};///<
    };
}
