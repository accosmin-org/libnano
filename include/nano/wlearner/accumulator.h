#pragma once

#include <nano/tensor.h>

namespace nano::wlearner
{
    ///
    /// \brief accumulates residuals & feature values of different moment orders
    ///     useful for fitting simple weak learners.
    ///
    class NANO_PUBLIC accumulator_t
    {
    public:
        explicit accumulator_t(const tensor3d_dims_t& tdims = make_dims(0, 0, 0));

        auto fvalues() const { return m_r1.size<0>(); }

        auto tdims() const { return make_dims(m_r1.size<1>(), m_r1.size<2>(), m_r1.size<3>()); }

        auto& x0(const tensor_size_t fv = 0) { return m_x0(fv); }

        auto& x1(const tensor_size_t fv = 0) { return m_x1(fv); }

        auto& x2(const tensor_size_t fv = 0) { return m_x2(fv); }

        auto r1(const tensor_size_t fv = 0) { return m_r1.array(fv); }

        auto rx(const tensor_size_t fv = 0) { return m_rx.array(fv); }

        auto r2(const tensor_size_t fv = 0) { return m_r2.array(fv); }

        auto x0(const tensor_size_t fv = 0) const { return m_x0(fv); }

        auto x1(const tensor_size_t fv = 0) const { return m_x1(fv); }

        auto x2(const tensor_size_t fv = 0) const { return m_x2(fv); }

        auto r1(const tensor_size_t fv = 0) const { return m_r1.array(fv); }

        auto rx(const tensor_size_t fv = 0) const { return m_rx.array(fv); }

        auto r2(const tensor_size_t fv = 0) const { return m_r2.array(fv); }

        void clear()
        {
            m_x0.zero();
            m_x1.zero();
            m_x2.zero();
            m_r1.zero();
            m_rx.zero();
            m_r2.zero();
        }

        void clear(const tensor_size_t fvalues)
        {
            m_x0.resize(fvalues);
            m_x1.resize(fvalues);
            m_x2.resize(fvalues);
            m_r1.resize(cat_dims(fvalues, tdims()));
            m_rx.resize(cat_dims(fvalues, tdims()));
            m_r2.resize(cat_dims(fvalues, tdims()));

            clear();
        }

        template <typename tarray>
        void update(tarray&& vgrad, const tensor_size_t fv = 0)
        {
            x0(fv) += 1;
            r1(fv) -= vgrad;
            r2(fv) += vgrad * vgrad;
        }

        template <typename tarray>
        void update(const scalar_t value, tarray&& vgrad, const tensor_size_t fv = 0)
        {
            update(vgrad, fv);
            x1(fv) += value;
            x2(fv) += value * value;
            rx(fv) -= vgrad * value;
        }

        ///
        /// \brief returns the (score, bin mapping) by selecting the k-best bins.
        ///
        /// NB: the x1 buffer is used to store score variations.
        ///
        std::tuple<scalar_t, indices_t> kbest(tensor_size_t kbest);

        ///
        /// \brief returns the (score, bin mapping) by clustering the bins in k-split parts.
        ///
        /// NB: the x0, r1, rx, r2 buffers are used to store the cluster statistics:
        ///     (count, first-order momentum, output, second-order momentum)!
        ///
        std::tuple<scalar_t, indices_t> ksplit(tensor_size_t ksplit);

    private:
        // attributes
        tensor1d_t m_x0{1}, m_x1{1}, m_x2{1}; ///<
        tensor4d_t m_r1, m_rx, m_r2;          ///<
    };
} // namespace nano::wlearner
