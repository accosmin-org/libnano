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

        auto bins() const { return m_r1.size<0>(); }

        auto tdims() const { return make_dims(m_r1.size<1>(), m_r1.size<2>(), m_r1.size<3>()); }

        auto& x0(const tensor_size_t bin = 0) { return m_x0(bin); }

        auto& x1(const tensor_size_t bin = 0) { return m_x1(bin); }

        auto& x2(const tensor_size_t bin = 0) { return m_x2(bin); }

        auto r1(const tensor_size_t bin = 0) { return m_r1.array(bin); }

        auto rx(const tensor_size_t bin = 0) { return m_rx.array(bin); }

        auto r2(const tensor_size_t bin = 0) { return m_r2.array(bin); }

        auto x0(const tensor_size_t bin = 0) const { return m_x0(bin); }

        auto x1(const tensor_size_t bin = 0) const { return m_x1(bin); }

        auto x2(const tensor_size_t bin = 0) const { return m_x2(bin); }

        auto r1(const tensor_size_t bin = 0) const { return m_r1.array(bin); }

        auto rx(const tensor_size_t bin = 0) const { return m_rx.array(bin); }

        auto r2(const tensor_size_t bin = 0) const { return m_r2.array(bin); }

        void clear();

        void clear(tensor_size_t bins);

        template <typename tarray>
        void update(tarray&& vgrad, const tensor_size_t bin = 0)
        {
            x0(bin) += 1;
            r1(bin) -= vgrad;
            r2(bin) += vgrad * vgrad;
        }

        template <typename tarray>
        void update(const scalar_t value, tarray&& vgrad, const tensor_size_t bin = 0)
        {
            update(vgrad, bin);
            x1(bin) += value;
            x2(bin) += value * value;
            rx(bin) -= vgrad * value;
        }

        std::vector<std::pair<scalar_t, tensor_size_t>> sort() const;

        std::tuple<tensor2d_t, tensor5d_t, tensor5d_t, tensor5d_t, tensor_mem_t<tensor_size_t, 2>> cluster() const;

        ///
        /// \brief return the constant that fits the given bin.
        ///
        auto fit_constant(const tensor_size_t bin) const { return r1(bin) / std::max(1.0, x0(bin)); }

        ///
        /// \brief return the residual sum of squares (RSS) obtained by predicting zero to the given bin.
        ///
        auto rss_zero(const tensor_size_t bin) const { return r2(bin).sum(); }

        ///
        /// \brief return the residual sum of squares (RSS) obtained by fitting a constant to the given bin.
        ///
        auto rss_constant(const tensor_size_t bin) const
        {
            return (r2(bin) - r1(bin) * r1(bin) / std::max(1.0, x0(bin))).sum();
        }

    private:
        // attributes
        tensor1d_t m_x0{1}; ///< sample count
        tensor1d_t m_x1{1}; ///< sum of feature values
        tensor1d_t m_x2{1}; ///< sum of squared feature values
        tensor4d_t m_r1;    ///< sum of gradients
        tensor4d_t m_rx;    ///< sum of feature value and gradient products
        tensor4d_t m_r2;    ///< sum of squared gradients
    };
} // namespace nano::wlearner
