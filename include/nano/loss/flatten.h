#pragma once

#include <cassert>
#include <nano/loss.h>
#include <nano/loss/class.h>
#include <nano/loss/error.h>

namespace nano
{
///
/// \brief un-structured loss function: the 3D structure of a sample is flatten
///     and all dimensions are considered the same in computing the loss.
///
/// NB: the multi-label classification problem is handled by summing or averaging:
///     - the associated binary classification loss value per output
///     - the associated 0-1 loss error per output
///
/// see the following resources regarding loss functions for classification:
///
/// (1): "On the design of robust classifiers for computer vision",
///      2010, by H. Masnadi-Shirazi, V. Mahadevan, N. Vasconcelos
///
/// (2): "On the design of loss functions for classification: theory, robustness to outliers, and SavageBoost",
///      2008, by H. Masnadi-Shirazi, N. Vasconcelos
///
template <class tloss>
class flatten_loss_t final : public loss_t
{
public:
    ///
    /// \brief constructor
    ///
    flatten_loss_t()
        : loss_t(string_t(tloss::prefix) + string_t(tloss::basename))
    {
        convex(tloss::convex);
        smooth(tloss::smooth);
    }

    ///
    /// \brief @see clonable_t
    ///
    rloss_t clone() const override { return std::make_unique<flatten_loss_t<tloss>>(*this); }

    ///
    /// \brief @see loss_t
    ///
    void error(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t errors) const override
    {
        assert(targets.dims() == outputs.dims());
        assert(errors.size() == targets.size<0>());

        for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++i)
        {
            errors(i) = tloss::error(targets.array(i), outputs.array(i));
        }
    }

    ///
    /// \brief @see loss_t
    ///
    void value(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t values) const override
    {
        assert(targets.dims() == outputs.dims());
        assert(values.size() == targets.size<0>());

        for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++i)
        {
            values(i) = tloss::value(targets.array(i), outputs.array(i));
        }
    }

    ///
    /// \brief @see loss_t
    ///
    void vgrad(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor4d_map_t vgrads) const override
    {
        assert(targets.dims() == vgrads.dims());
        assert(targets.dims() == outputs.dims());

        for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++i)
        {
            tloss::vgrad(targets.array(i), outputs.array(i), vgrads.array(i));
        }
    }

    ///
    /// \brief @see loss_t
    ///
    void vhess(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor3d_map_t vhesss) const override
    {
        assert(targets.size<0>() == vhesss.size<0>());
        assert(targets.size<1>() * targets.size<2>() * targets.size<3>() == vhesss.size<1>());
        assert(targets.size<1>() * targets.size<2>() * targets.size<3>() == vhesss.size<2>());
        assert(targets.dims() == outputs.dims());

        for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++i)
        {
            tloss::vhess(targets.array(i), outputs.array(i), vhesss.tensor(i));
        }
    }
};

namespace detail
{
///
/// \brief class negative log-likelihood loss (also called cross-entropy loss).
///
template <class terror>
struct classnll_t : public terror
{
    static constexpr auto convex   = true;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "classnll";

    template <class tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        tensor_size_t imax = 0;
        const auto    omax = output.maxCoeff(&imax);

        scalar_t value = std::numeric_limits<scalar_t>::epsilon();
        scalar_t posum = 0;
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            value += std::exp(output(i) - omax);
            if (is_pos_target(target(i)))
            {
                posum += output(i);
            }
        }
        return std::log(value) - posum + omax;
    }

    template <class tarray, class tgarray>
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        tensor_size_t imax = 0;
        const auto    omax = output.maxCoeff(&imax);

        scalar_t value = 0;
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            value += (vgrad(i) = std::exp(output(i) - omax));
        }
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            vgrad(i) /= value;
            if (is_pos_target(target(i)))
            {
                vgrad(i) -= 1.0;
            }
        }
    }

    template <class tarray, class thmatrix>
    static void vhess(const tarray& target, const tarray& output, thmatrix vhess)
    {
        vhess.full(0.0);
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            const auto x = -target(i) * output(i);
            const auto h = (x < 1.0) ? (std::exp(x) * (1.0 - std::exp(x)) / square(1.0 + std::exp(x)))
                                     : ((std::exp(-x) - 1) / square(1.0 + std::exp(-x)));
            vhess(i, i)  = target(i) * target(i) * h;
        }
    }
};

///
/// \brief multi-class exponential loss.
///
template <class terror>
struct exponential_t : public terror
{
    static constexpr auto convex   = true;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "exponential";

    template <class tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        return (-target * output).exp().sum();
    }

    template <class tarray, class tgarray>
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        vgrad = -target * (-target * output).exp();
    }

    template <class tarray, class thmatrix>
    static void vhess(const tarray& target, const tarray& output, thmatrix vhess)
    {
        vhess = (-target * output).exp().matrix().asDiagonal();
    }
};

///
/// \brief multi-class logistic loss.
///
template <class terror>
struct logistic_t : public terror
{
    static constexpr auto convex   = true;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "logistic";

    template <class tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        scalar_t value = 0.0;
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            const auto x = -target(i) * output(i);
            value += (x < 1.0) ? std::log1p(std::exp(x)) : (x + std::log1p(std::exp(-x)));
        }
        return value;
    }

    template <class tarray, class tgarray>
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            const auto x = -target(i) * output(i);
            const auto g = (x < 1.0) ? (std::exp(x) / (1.0 + std::exp(x))) : (1.0 / (1.0 + std::exp(-x)));
            vgrad(i)     = -target(i) * g;
        }
    }

    template <class tarray, class thmatrix>
    static void vhess(const tarray& target, const tarray& output, thmatrix vhess)
    {
        vhess.full(0.0);
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            const auto x = -target(i) * output(i);
            const auto h = (x < 1.0) ? (std::exp(x) * (1.0 - std::exp(x)) / square(1.0 + std::exp(x)))
                                     : ((std::exp(-x) - 1) / square(1.0 + std::exp(-x)));
            vhess(i, i)  = target(i) * target(i) * h;
        }
    }
};

///
/// \brief multi-class hinge loss.
///
template <class terror>
struct hinge_t : public terror
{
    static constexpr auto convex   = true;
    static constexpr auto smooth   = false;
    static constexpr auto basename = "hinge";

    template <class tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        return (1.0 - target * output).max(0).sum();
    }

    template <class tarray, class tgarray>
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        vgrad = -target * ((1.0 - target * output).sign() + 1.0) * 0.5;
    }

    template <class tarray, class thmatrix>
    static void vhess([[maybe_unused]] const tarray& target, [[maybe_unused]] const tarray& output, thmatrix vhess)
    {
        assert(false);
        vhess.full(0.0);
    }
};

///
/// \brief multi-class squared hinge loss (smooth version of the hinge loss).
///
template <class terror>
struct squared_hinge_t : public terror
{
    static constexpr auto convex   = true;
    static constexpr auto smooth   = false;
    static constexpr auto basename = "squared-hinge";

    template <class tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        return (1.0 - target * output).max(0).square().sum();
    }

    template <class tarray, class tgarray>
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        vgrad = -target * (1.0 - target * output).max(0) * 2.0;
    }

    template <class tarray, class thmatrix>
    static void vhess([[maybe_unused]] const tarray& target, [[maybe_unused]] const tarray& output, thmatrix vhess)
    {
        assert(false);
        vhess.full(0.0);
    }
};

///
/// \brief multi-class savage loss.
///
template <class terror>
struct savage_t : public terror
{
    static constexpr auto convex   = false;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "savage";

    template <class tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        const auto edges = (target * output).exp();

        return (1.0 / (1.0 + edges).square()).sum();
    }

    template <class tarray, class tgarray>
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        const auto edges = (target * output).exp();

        vgrad = -2.0 * target / (1.0 + edges).cube();
    }

    template <class tarray, class thmatrix>
    static void vhess(const tarray& target, const tarray& output, thmatrix vhess)
    {
        const auto edges = (target * output).exp();

        vhess = -2.0 * (edges - 2.0 * edges.square()) / (1.0 + edges).square().square();
    }
};

///
/// \brief multi-class tangent loss.
///
template <class terror>
struct tangent_t : public terror
{
    static constexpr auto convex   = false;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "tangent";

    template <class tarray>
    static auto atan(const tarray& target, const tarray& output)
    {
        return 2.0 * (target * output).atan() - 1.0;
    }

    template <class tarray>
    static auto gdiv(const tarray& target, const tarray& output)
    {
        return 1.0 + (target * output).square();
    }

    template <class tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        return atan(target, output).square().sum();
    }

    template <class tarray, class tgarray>
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        vgrad = 4.0 * target * atan(target, output) / gdiv(target, output);
    }

    template <class tarray, class thmatrix>
    static void vhess(const tarray& target, const tarray& output, thmatrix vhess)
    {
        vhess = (8.0 * target * (1.0 - output * atan(target, output)) / gdiv(target, output).square())
                    .matrix()
                    .asDiagonal();
    }
};

///
/// \brief absolute-difference loss.
///
template <class terror>
struct mae_t : public terror
{
    static constexpr auto convex   = true;
    static constexpr auto smooth   = false;
    static constexpr auto basename = "mae";

    template <class tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        return (output - target).abs().sum();
    }

    template <class tarray, class tgarray>
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        vgrad = (output - target).sign();
    }

    template <class tarray, class thmatrix>
    static void vhess([[maybe_unused]] const tarray& target, [[maybe_unused]] const tarray& output, thmatrix vhess)
    {
        assert(false);
        vhess.full(0.0);
    }
};

///
/// \brief squared-difference loss.
///
template <class terror>
struct mse_t : public terror
{
    static constexpr auto convex   = true;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "mse";

    template <class tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        return 0.5 * (output - target).square().sum();
    }

    template <class tarray, class tgarray>
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        vgrad = output - target;
    }

    template <class tarray, class thmatrix>
    static void vhess([[maybe_unused]] const tarray& target, [[maybe_unused]] const tarray& output, thmatrix vhess)
    {
        vhess = matrix_t::identity(vhess.rows(), vhess.cols());
    }
};

///
/// \brief robust-to-noise Cauchy loss.
///
template <class terror>
struct cauchy_t : public terror
{
    static constexpr auto convex   = false;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "cauchy";

    template <class tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        return 0.5 * ((target - output).square() + 1).log().sum();
    }

    template <class tarray, class tgarray>
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        vgrad = (output - target) / (1 + (output - target).square());
    }

    template <class tarray, class thmatrix>
    static void vhess(const tarray& target, const tarray& output, thmatrix vhess)
    {
        vhess = ((output - target) / (1 + (output - target).square()).square()).matrix().asDiagonal();
    }
};
} // namespace detail

using mae_loss_t    = flatten_loss_t<detail::mae_t<loss::detail::absdiff_t>>;
using mse_loss_t    = flatten_loss_t<detail::mse_t<loss::detail::absdiff_t>>;
using cauchy_loss_t = flatten_loss_t<detail::cauchy_t<loss::detail::absdiff_t>>;

using shinge_loss_t         = flatten_loss_t<detail::hinge_t<loss::detail::sclass_t>>;
using ssavage_loss_t        = flatten_loss_t<detail::savage_t<loss::detail::sclass_t>>;
using stangent_loss_t       = flatten_loss_t<detail::tangent_t<loss::detail::sclass_t>>;
using sclassnll_loss_t      = flatten_loss_t<detail::classnll_t<loss::detail::sclass_t>>;
using slogistic_loss_t      = flatten_loss_t<detail::logistic_t<loss::detail::sclass_t>>;
using sexponential_loss_t   = flatten_loss_t<detail::exponential_t<loss::detail::sclass_t>>;
using ssquared_hinge_loss_t = flatten_loss_t<detail::squared_hinge_t<loss::detail::sclass_t>>;

using mhinge_loss_t         = flatten_loss_t<detail::hinge_t<loss::detail::mclass_t>>;
using msavage_loss_t        = flatten_loss_t<detail::savage_t<loss::detail::mclass_t>>;
using mtangent_loss_t       = flatten_loss_t<detail::tangent_t<loss::detail::mclass_t>>;
using mlogistic_loss_t      = flatten_loss_t<detail::logistic_t<loss::detail::mclass_t>>;
using mexponential_loss_t   = flatten_loss_t<detail::exponential_t<loss::detail::mclass_t>>;
using msquared_hinge_loss_t = flatten_loss_t<detail::squared_hinge_t<loss::detail::mclass_t>>;
} // namespace nano
