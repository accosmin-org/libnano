#pragma once

#include <cassert>
#include <nano/loss.h>
#include <nano/mlearn.h>

namespace nano
{
    ///
    /// \brief multi-class classification loss that predicts the labels with positive output.
    ///
    template <typename top>
    class mclassification_loss_t final : public loss_t
    {
    public:

        ///
        /// \brief @see loss_t
        ///
        scalar_t error(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const override
        {
            assert(target.dims() == output.dims());

            const auto edges = target.array() * output.array();
            const auto epsilon = std::numeric_limits<scalar_t>::epsilon();
            return static_cast<scalar_t>((edges < epsilon).count());
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

    ///
    /// \brief single-class classification loss that predicts the label with the highest score.
    ///
    template <typename top>
    class sclassification_loss_t final : public loss_t
    {
    public:

        ///
        /// \brief @see loss_t
        ///
        scalar_t error(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const override
        {
            assert(target.dims() == output.dims());

            if (target.size() > 1)
            {
                tensor_size_t idx;
                output.array().maxCoeff(&idx);

                return is_pos_target(target(idx)) ? 0 : 1;
            }
            else
            {
                const auto edges = target.array() * output.array();
                const auto epsilon = std::numeric_limits<scalar_t>::epsilon();
                return static_cast<scalar_t>((edges < epsilon).count());
            }
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
        /// \brief class negative log-likelihood loss (also called cross-entropy loss).
        ///
        struct classnll_t
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return  std::log(output.exp().sum()) -
                        std::log(((1 + target) * output.exp()).sum() / 2);
            }

            template <typename tarray>
            static auto vgrad(const tarray& target, const tarray& output)
            {
                return  output.exp() / output.exp().sum() -
                        (1 + target) * output.exp() / ((1 + target) * output.exp()).sum();
            }
        };

        ///
        /// \brief multi-class exponential loss.
        ///
        struct exponential_t
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return (-target * output).exp().sum();
            }

            template <typename tarray>
            static auto vgrad(const tarray& target, const tarray& output)
            {
                return -target * (-target * output).exp();
            }
        };

        ///
        /// \brief multi-class logistic loss.
        ///
        struct logistic_t
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return  (1 + (-target * output).exp()).log().sum();
            }

            template <typename tarray>
            static auto vgrad(const tarray& target, const tarray& output)
            {
                return  -target * (-target * output).exp() /
                        (1 + (-target * output).exp());
            }
        };

        ///
        /// \brief multi-class hinge loss.
        ///
        struct hinge_t
        {
            template <typename tarray>
            static auto value(const tarray& target, const tarray& output)
            {
                return (1 - target * output).max(0).sum();
            }

            template <typename tarray>
            static auto vgrad(const tarray& target, const tarray& output)
            {
                return -target * (1 - target * output).sign();
            }
        };
    }

    using shinge_loss_t = sclassification_loss_t<detail::hinge_t>;
    using sclassnll_loss_t = sclassification_loss_t<detail::classnll_t>;
    using slogistic_loss_t = sclassification_loss_t<detail::logistic_t>;
    using sexponential_loss_t = sclassification_loss_t<detail::exponential_t>;

    using mhinge_loss_t = mclassification_loss_t<detail::hinge_t>;
    using mlogistic_loss_t = mclassification_loss_t<detail::logistic_t>;
    using mexponential_loss_t = mclassification_loss_t<detail::exponential_t>;
}
