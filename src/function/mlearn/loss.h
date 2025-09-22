#pragma once

#include <function/mlearn/linear.h>

namespace nano
{
///
/// \brief mean-squared-error (MSE) loss.
///
class NANO_PUBLIC loss_mse_t
{
public:
    static constexpr auto convex     = true;
    static constexpr auto smooth     = true;
    static constexpr auto basename   = "mse";
    static constexpr auto regression = true;

    static scalar_t fx(matrix_cmap_t outputs, matrix_cmap_t targets);
    static void     gx(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx);
    static void     hx(matrix_cmap_t outputs, matrix_cmap_t targets, tensor3d_map_t Hx);
};

///
/// \brief mean-absolute-error (MAE) loss.
///
class NANO_PUBLIC loss_mae_t
{
public:
    static constexpr auto convex     = true;
    static constexpr auto smooth     = false;
    static constexpr auto basename   = "mae";
    static constexpr auto regression = true;

    static scalar_t fx(matrix_cmap_t outputs, matrix_cmap_t targets);
    static void     gx(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx);
    static void     hx(matrix_cmap_t outputs, matrix_cmap_t targets, tensor3d_map_t Hx);
};

///
/// \brief cauchy loss.
///
class NANO_PUBLIC loss_cauchy_t
{
public:
    static constexpr auto convex     = false;
    static constexpr auto smooth     = false;
    static constexpr auto basename   = "cauchy";
    static constexpr auto regression = true;

    static scalar_t fx(matrix_cmap_t outputs, matrix_cmap_t targets);
    static void     gx(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx);
    static void     hx(matrix_cmap_t outputs, matrix_cmap_t targets, tensor3d_map_t Hx);
};

///
/// \brief hinge loss (linear SVM).
///
class NANO_PUBLIC loss_hinge_t
{
public:
    static constexpr auto convex     = true;
    static constexpr auto smooth     = false;
    static constexpr auto basename   = "hinge";
    static constexpr auto regression = false;

    static scalar_t fx(matrix_cmap_t outputs, matrix_cmap_t targets);
    static void     gx(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx);
    static void     hx(matrix_cmap_t outputs, matrix_cmap_t targets, tensor3d_map_t Hx);
};

///
/// \brief logistic loss (binary classification).
///
class NANO_PUBLIC loss_logistic_t
{
public:
    static constexpr auto convex     = true;
    static constexpr auto smooth     = true;
    static constexpr auto basename   = "logistic";
    static constexpr auto regression = false;

    static scalar_t fx(matrix_cmap_t outputs, matrix_cmap_t targets);
    static void     gx(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx);
    static void     hx(matrix_cmap_t outputs, matrix_cmap_t targets, tensor3d_map_t Hx);
};
} // namespace nano
