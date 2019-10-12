#pragma once

#include <cassert>
#include <nano/loss.h>

namespace nano
{
    ///
    /// \brief generic (multivariate) regression loss that upper-bounds
    ///     the L1-distance between target and output.
    ///
    template <typename top>
    class regression_loss_t final : public loss_t
    {
    public:

        ///
        /// \brief @see loss_t
        ///
        scalar_t error(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const override
        {
            assert(target.dims() == output.dims());

            return (target.array() - output.array()).abs().sum();
        }

        ///
        /// \brief @see loss_t
        ///
        scalar_t value(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const override
        {
            assert(target.dims() == output.dims());

            return top::value(target.array(), output.array());
        }

        ///
        /// \brief @see loss_t
        ///
        void vgrad(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output,
            const tensor3d_map_t& vgrad) const override
        {
            assert(target.dims() == output.dims());
            assert(target.dims() == vgrad.dims());

            vgrad.array() = top::vgrad(target.array(), output.array());
        }
    };

    namespace detail
    {
        ///
        /// \brief absolute-difference loss.
        ///
        struct absolute_t
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return (output - target).abs().sum();
            }

            template <typename tarray>
            static auto vgrad(const tarray& target, const tarray& output)
            {
                return (output - target).sign();
            }
        };

        ///
        /// \brief squared-difference loss.
        ///
        struct squared_t
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return scalar_t(0.5) * (output - target).square().sum();
            }

            template <typename tarray>
            static auto vgrad(const tarray& target, const tarray& output)
            {
                return output - target;
            }
        };

        ///
        /// \brief robust-to-noise Cauchy loss.
        ///
        struct cauchy_t
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return scalar_t(0.5) * ((target - output).square() + 1).log().sum();
            }

            template <typename tarray>
            static auto vgrad(const tarray& target, const tarray& output)
            {
                return (output - target) / (1 + (output - target).square());
            }
        };
    }

    using cauchy_loss_t = regression_loss_t<detail::cauchy_t>;
    using squared_loss_t = regression_loss_t<detail::squared_t>;
    using absolute_loss_t = regression_loss_t<detail::absolute_t>;
}
