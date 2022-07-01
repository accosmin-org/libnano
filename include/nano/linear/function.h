#pragma once

#include <nano/function.h>
#include <nano/generator/iterator.h>
#include <nano/linear/cache.h>
#include <nano/loss.h>

namespace nano::linear
{
    ///
    /// \brief the ERM criterion used for optimizing the parameters of a linear model,
    ///     using a given loss function.
    ///
    /// NB: the ERM loss can be optionally regularized by penalizing:
    ///     - (1) the L1-norm of the weights matrix - like in LASSO
    ///     - (2) the L2-norm of the weights matrix - like in RIDGE
    ///     - (3) both the L1 and the L2-norms of the weights matrix - like in elastic net regularization
    ///     - (4) the variance of the loss values - like in VadaBoost
    ///
    class NANO_PUBLIC function_t final : public ::nano::function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        function_t(flatten_iterator_t, const loss_t&, scalar_t l1reg, scalar_t l2reg, scalar_t vAreg);

        ///
        /// \brief extract the weight matrix from the given tensor
        ///
        template <typename ttensor>
        auto weights(ttensor& x) const
        {
            assert(x.size() == m_isize * m_tsize + m_tsize);
            return map_tensor(x.data(), m_tsize, m_isize);
        }

        template <typename ttensor>
        auto weights(const ttensor& x) const
        {
            assert(x.size() == m_isize * m_tsize + m_tsize);
            return map_tensor(x.data(), m_tsize, m_isize);
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
        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr, vgrad_config_t = vgrad_config_t{}) const override;

        ///
        /// \brief access functions
        ///
        const auto& loss() const { return m_loss; }

        const auto& iterator() const { return m_iterator; }

    private:
        using caches_t = std::vector<cache_t>;

        // attributes
        flatten_iterator_t m_iterator;   ///<
        const loss_t&      m_loss;       ///<
        scalar_t           m_l1reg{0.0}; ///< regularization factor - see (1), (3)
        scalar_t           m_l2reg{0.0}; ///< regularization factor - see (2), (3)
        scalar_t           m_vAreg{0.0}; ///< regularization factor - see (4)
        tensor_size_t      m_isize{0};   ///< #inputs (e.g. size of the flatten input feature tensor)
        tensor_size_t      m_tsize{0};   ///< #targets (e.g. size of the flatten target tensor, number of classes)
        mutable caches_t   m_caches;     ///< liner model-specific buffers per thread
    };
} // namespace nano::linear
