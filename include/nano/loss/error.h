#pragma once

#include <nano/tensor.h>

namespace nano::loss::detail
{
///
/// \brief error measure for unstructured (multivariate) regression:
///     - the L1-distance between target and output.
///
struct absdiff_t
{
    static constexpr auto prefix = "";

    template <class tarray>
    requires is_eigen_v<tarray>
    static auto error(const tarray& target, const tarray& output)
    {
        return (target - output).abs().sum();
    }
};

///
/// \brief error measure for multi-class classification:
///     - the number of mis-matched predictions where the label is predicted if its output is positive.
///
struct mclass_t
{
    static constexpr auto prefix = "m-";

    template <class tarray>
    requires is_eigen_v<tarray>
    static auto error(const tarray& target, const tarray& output)
    {
        const auto     edges   = target * output;
        constexpr auto epsilon = std::numeric_limits<scalar_t>::epsilon();
        return static_cast<scalar_t>((edges < epsilon).count());
    }
};

///
/// \brief error measure for single-class classification:
///     - 0-1 loss where the predicted label is the one with the highest score (if multi-class) or positive (if
///     binary).
///
struct sclass_t
{
    static constexpr auto prefix = "s-";

    template <class tarray>
    requires is_eigen_v<tarray>
    static auto error(const tarray& target, const tarray& output)
    {
        if (target.size() > 1)
        {
            // multi-class
            tensor_size_t idx = -1;
            output.array().maxCoeff(&idx);

            return static_cast<scalar_t>(is_pos_target(target(idx)) ? 0 : 1);
        }
        else
        {
            // binary classification
            const auto     edges   = target.array() * output.array();
            constexpr auto epsilon = std::numeric_limits<scalar_t>::epsilon();
            return static_cast<scalar_t>((edges < epsilon).count());
        }
    }
};
} // namespace nano::loss::detail
