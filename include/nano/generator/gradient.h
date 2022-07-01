#pragma once

#include <nano/string.h>
#include <nano/tensor/tensor.h>

namespace nano
{
    ///
    /// \brief supported symmetric 3x3 kernels.
    ///
    enum class kernel3x3_type
    {
        sobel,
        scharr,
        prewitt,
    };

    template <>
    inline enum_map_t<kernel3x3_type> enum_string<kernel3x3_type>()
    {
        return {
            {  kernel3x3_type::sobel,   "sobel"},
            { kernel3x3_type::scharr,  "scharr"},
            {kernel3x3_type::prewitt, "prewitt"},
        };
    }

    inline std::ostream& operator<<(std::ostream& stream, kernel3x3_type value) { return stream << scat(value); }

    ///
    /// \brief construct symmetric 3x3 kernels for computing image gradients.
    ///
    template <typename tscalar>
    std::array<tscalar, 3> make_kernel3x3(kernel3x3_type type)
    {
        switch (type)
        {
        case kernel3x3_type::sobel:
            return {static_cast<tscalar>(1) / static_cast<tscalar>(4),
                    static_cast<tscalar>(2) / static_cast<tscalar>(4),
                    static_cast<tscalar>(1) / static_cast<tscalar>(4)};

        case kernel3x3_type::scharr:
            return {static_cast<tscalar>(3) / static_cast<tscalar>(16),
                    static_cast<tscalar>(10) / static_cast<tscalar>(16),
                    static_cast<tscalar>(3) / static_cast<tscalar>(16)};

        case kernel3x3_type::prewitt:
            return {static_cast<tscalar>(1) / static_cast<tscalar>(3),
                    static_cast<tscalar>(1) / static_cast<tscalar>(3),
                    static_cast<tscalar>(1) / static_cast<tscalar>(3)};

        default:
            return {std::numeric_limits<tscalar>::quiet_NaN(), std::numeric_limits<tscalar>::quiet_NaN(),
                    std::numeric_limits<tscalar>::quiet_NaN()};
        }
    }

    ///
    /// \brief computation mode for gradients.
    ///
    enum class gradient3x3_mode
    {
        gradx,     ///< horizontal gradient
        grady,     ///< vertical gradient
        magnitude, ///< edge magnitude
        angle      ///< edge orientation
    };

    ///
    /// \brief compute for each pixel the horizontal/vertical gradients, the edge magnitude and the edge orientation
    ///     in a 2D image using a symmetric 3x3 kernel.
    ///
    /// NB: optionally the pixels values can be scaled, for example by standardizing them across the image.
    ///
    template <typename tscalar_input, typename tscalar_output>
    void gradient3x3(gradient3x3_mode mode, tensor_cmap_t<tscalar_input, 2> input,
                     const std::array<tscalar_output, 3> kernel, tensor_map_t<tscalar_output, 2> output)
    {
        const auto rows = output.template size<0>();
        const auto cols = output.template size<1>();

        assert(input.template size<0>() == rows + 2);
        assert(input.template size<1>() == cols + 2);

        const auto make_gg = [&](tscalar_input v0, tscalar_input v1, tscalar_input v2, tscalar_input v3,
                                 tscalar_input v4, tscalar_input v5)
        {
            const auto d0 = static_cast<tscalar_output>(v0) - static_cast<tscalar_output>(v1);
            const auto d1 = static_cast<tscalar_output>(v2) - static_cast<tscalar_output>(v3);
            const auto d2 = static_cast<tscalar_output>(v4) - static_cast<tscalar_output>(v5);

            return kernel[0] * d0 + kernel[1] * d1 + kernel[2] * d2;
        };

        const auto make_gx = [&](tensor_size_t row, tensor_size_t col)
        {
            return make_gg(input(row + 0, col + 2), input(row + 0, col), input(row + 1, col + 2), input(row + 1, col),
                           input(row + 2, col + 2), input(row + 2, col));
        };

        const auto make_gy = [&](tensor_size_t row, tensor_size_t col)
        {
            return make_gg(input(row + 2, col + 0), input(row, col + 0), input(row + 2, col + 1), input(row, col + 1),
                           input(row + 2, col + 2), input(row, col + 2));
        };

        switch (mode)
        {
        case gradient3x3_mode::gradx:
            for (tensor_size_t row = 0; row < rows; ++row)
            {
                for (tensor_size_t col = 0; col < cols; ++col)
                {
                    const auto gx = make_gx(row, col);

                    output(row, col) = gx;
                }
            }
            break;

        case gradient3x3_mode::grady:
            for (tensor_size_t row = 0; row < rows; ++row)
            {
                for (tensor_size_t col = 0; col < cols; ++col)
                {
                    const auto gy = make_gy(row, col);

                    output(row, col) = gy;
                }
            }
            break;

        case gradient3x3_mode::magnitude:
            for (tensor_size_t row = 0; row < rows; ++row)
            {
                for (tensor_size_t col = 0; col < cols; ++col)
                {
                    const auto gx = make_gx(row, col);
                    const auto gy = make_gy(row, col);

                    output(row, col) = std::sqrt(gx * gx + gy * gy);
                }
            }
            break;

        default:
            for (tensor_size_t row = 0; row < rows; ++row)
            {
                for (tensor_size_t col = 0; col < cols; ++col)
                {
                    const auto gx = make_gx(row, col);
                    const auto gy = make_gy(row, col);

                    output(row, col) = std::atan2(gy, gx);
                }
            }
        }
    }
} // namespace nano
