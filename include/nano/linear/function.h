#pragma once

#include <nano/dataset/iterator.h>
#include <nano/function.h>
#include <nano/linear/accumulator.h>
#include <nano/loss.h>

namespace nano::linear
{
///
/// \brief the empirical risk minimization (ERM) criterion
///     used for optimizing the parameters of a linear model with a generic loss function.
///
/// NB: the ERM loss can be optionally regularized by penalizing:
///     - (1) the L1-norm of the weights matrix - like in LASSO
///     - (2) the L2-norm of the weights matrix - like in RIDGE
///     - (3) both the L1 and the L2-norms of the weights matrix - like in elastic net regularization
///
class NANO_PUBLIC function_t final : public ::nano::function_t
{
public:
    ///
    /// \brief constructor
    ///
    function_t(const flatten_iterator_t&, const loss_t&, scalar_t l1reg, scalar_t l2reg);

    ///
    /// \brief extract the weight matrix from the given tensor
    ///
    template <class ttensor>
    auto weights(ttensor& x) const
    {
        assert(x.size() == m_isize * m_tsize + m_tsize);
        return map_tensor(x.data(), m_tsize, m_isize);
    }

    template <class ttensor>
    auto weights(const ttensor& x) const
    {
        assert(x.size() == m_isize * m_tsize + m_tsize);
        return map_tensor(x.data(), m_tsize, m_isize);
    }

    ///
    /// \brief extract the bias vector from the given tensor
    ///
    template <class ttensor>
    auto bias(ttensor& x) const
    {
        assert(x.size() == m_isize * m_tsize + m_tsize);
        return map_tensor(x.data() + m_isize * m_tsize, m_tsize);
    }

    template <class ttensor>
    auto bias(const ttensor& x) const
    {
        assert(x.size() == m_isize * m_tsize + m_tsize);
        return map_tensor(x.data() + m_isize * m_tsize, m_tsize);
    }

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override;

private:
    // attributes
    const flatten_iterator_t& m_iterator;     ///<
    const loss_t&             m_loss;         ///<
    scalar_t                  m_l1reg{0.0};   ///< regularization factor - see (1), (3)
    scalar_t                  m_l2reg{0.0};   ///< regularization factor - see (2), (3)
    tensor_size_t             m_isize{0};     ///< #inputs (e.g. size of the flatten input feature tensor)
    tensor_size_t             m_tsize{0};     ///< #targets (e.g. size of the flatten target tensor, number of classes)
    mutable accumulators_t    m_accumulators; ///< liner model-specific buffers per thread
};
} // namespace nano::linear
