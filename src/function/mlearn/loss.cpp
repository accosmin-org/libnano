#include <function/mlearn/loss.h>

using namespace nano;

scalar_t loss_mse_t::fx(matrix_cmap_t outputs, matrix_cmap_t targets)
{
    const auto samples = outputs.rows();
    const auto delta   = outputs - targets;

    return 0.5 * delta.squaredNorm() / static_cast<scalar_t>(samples);
}

void loss_mse_t::gx(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx)
{
    const auto delta = outputs - targets;

    gx = delta;
}

void loss_mse_t::hx(matrix_cmap_t outputs, matrix_cmap_t, tensor3d_map_t Hx)
{
    const auto [samples, osize] = outputs.dims();

    for (tensor_size_t sample = 0; sample < samples; ++sample)
    {
        Hx.tensor(sample) = matrix_t::identity(osize, osize);
    }
}

scalar_t loss_mae_t::fx(matrix_cmap_t outputs, matrix_cmap_t targets)
{
    const auto samples = outputs.rows();
    const auto delta   = outputs - targets;

    return delta.array().abs().sum() / static_cast<scalar_t>(samples);
}

void loss_mae_t::gx(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx)
{
    const auto delta = outputs - targets;

    gx = delta.array().sign().matrix();
}

void loss_mae_t::hx(matrix_cmap_t, matrix_cmap_t, tensor3d_map_t Hx)
{
    Hx.full(0.0);
}

scalar_t loss_cauchy_t::fx(matrix_cmap_t outputs, matrix_cmap_t targets)
{
    const auto samples = outputs.rows();
    const auto delta   = outputs - targets;

    return (delta.array().square() + 1.0).log().sum() / static_cast<scalar_t>(samples);
}

void loss_cauchy_t::gx(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx)
{
    const auto delta = outputs - targets;

    gx = (2.0 * delta.array() / (1.0 + delta.array().square())).matrix();
}

void loss_cauchy_t::hx(matrix_cmap_t outputs, matrix_cmap_t targets, tensor3d_map_t Hx)
{
    const auto samples = outputs.rows();

    for (tensor_size_t sample = 0; sample < samples; ++sample)
    {
        const auto odelta = outputs.array(sample) - targets.array(sample);

        Hx.tensor(sample) = (2.0 * (1.0 - odelta.square()) / (1.0 + odelta.square()).square()).matrix().asDiagonal();
    }
}

scalar_t loss_hinge_t::fx(matrix_cmap_t outputs, matrix_cmap_t targets)
{
    const auto samples = outputs.rows();
    const auto edges   = -outputs.array() * targets.array();

    return (1.0 + edges).max(0.0).sum() / static_cast<scalar_t>(samples);
}

void loss_hinge_t::gx(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx)
{
    const auto edges = -outputs.array() * targets.array();

    gx = (-targets.array() * ((1.0 + edges).sign() * 0.5 + 0.5)).matrix();
}

void loss_hinge_t::hx(matrix_cmap_t, matrix_cmap_t, tensor3d_map_t Hx)
{
    Hx.full(0.0);
}

scalar_t loss_logistic_t::fx(matrix_cmap_t outputs, matrix_cmap_t targets)
{
    const auto samples = outputs.rows();
    const auto edges   = (-outputs.array() * targets.array()).exp();

    return (1.0 + edges).log().sum() / static_cast<scalar_t>(samples);
}

void loss_logistic_t::gx(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx)
{
    const auto edges = (-outputs.array() * targets.array()).exp();

    gx = ((-targets.array() * edges) / (1.0 + edges)).matrix();
}

void loss_logistic_t::hx(matrix_cmap_t outputs, matrix_cmap_t targets, tensor3d_map_t Hx)
{
    const auto samples = outputs.rows();

    for (tensor_size_t sample = 0; sample < samples; ++sample)
    {
        const auto oedge = (-outputs.array(sample) * targets.array(sample)).exp();

        Hx.tensor(sample) = (oedge / (1.0 + oedge).square()).matrix().asDiagonal();
    }
}
