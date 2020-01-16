#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief models the learning rate schedule of the form:
    ///     `lrate_k = lrate0/(1 + k)^decay` afterwards, where k is the current update.
    ///
    /// NB: setting decay to zero amounts to using a constant learning rate.
    ///
    class lrate_schedule_t
    {
    public:

        ///
        /// \brief constructor
        ///
        lrate_schedule_t(const scalar_t lrate0, const scalar_t decay) :
            m_lrate0(lrate0),
            m_decay(decay)
        {
        }

        ///
        /// \brief returns the current update step
        ///
        auto k() const
        {
            return m_k;
        }

        ///
        /// \brief returns the current learning step
        ///
        auto get() const
        {
            return m_lrate0 / std::pow(m_k + 1.0, m_decay);
        }

        ///
        /// \brief move to the next update step
        ///
        auto& operator++()
        {
            m_k += 1.0;
            return *this;
        }

    private:

        // attributes
        scalar_t        m_lrate0{1};        ///< initial learning rate
        scalar_t        m_k{0};             ///< update index
        scalar_t        m_decay{1};         ///< decay factor
    };

    ///
    /// \brief models the minibatch size update, potentially increasing it geometrically.
    ///
    /// NB: setting the ratio to one amounts to using a constant minibatch size.
    ///
    class batch_schedule_t
    {
    public:

        ///
        /// \brief constructor
        ///
        batch_schedule_t(const tensor_size_t batch0, const scalar_t batchr, const function_t& function) :
            m_function(function),
            m_batch(static_cast<scalar_t>(batch0)),
            m_batchr(batchr),
            m_batchM(std::min(100 * batch0, function.summands()))
        {
        }

        ///
        /// \brief returns the current minibatch size (~number of summands)
        ///
        auto get() const
        {
            return std::min(static_cast<tensor_size_t>(m_batch), m_batchM);
        }

        ///
        /// \brief move to the next update step
        ///
        auto& operator++()
        {
            m_batch *= m_batchr;
            return *this;
        }

        ///
        /// \brief loop over all summands using the given operator called with [begin, end) range of summands to use.
        ///
        /// NB: this corresponds to one epoch in machine learning.
        /// NB: the learning rate schedule is updated accordingly.
        ///
        template <typename toperator>
        void loop(lrate_schedule_t& lrate, const toperator& op)
        {
            m_function.shuffle();
            for (tensor_size_t begin = 0; begin + get() <= m_function.summands(); begin += get(), ++ *this, ++ lrate)
            {
                if (!op(begin, begin + get(), lrate.get()))
                {
                    break;
                }
            }
        }

    private:

        // attributes
        const function_t&   m_function;         ///<
        scalar_t            m_batch{1};         ///< current minibatch size in number of summands
        scalar_t            m_batchr{1};        ///< minibatch ratio to geometrically increase the minibatch size
        tensor_size_t       m_batchM{1024};     ///< maximum minibatch size in number of summands
    };
}
