#include <function/mlearn/loss.h>

using namespace nano;

scalar_t loss_mse_t::eval(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx, tensor3d_map_t Hx)
{
    const auto delta = outputs - targets;

    gx = delta;

    for (tensor_size_t sample = 0, samples = outputs.rows(), osize = outputs.cols(); sample < samples; ++sample)
    {
        Hx.tensor(sample) = matrix_t::identity(osize, osize);
    }

    return 0.5 * delta.squaredNorm() / static_cast<scalar_t>(outputs.rows());
}

scalar_t loss_mae_t::eval(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx, tensor3d_map_t Hx)
{
    const auto delta = outputs - targets;

    gx = delta.array().sign().matrix();

    Hx.full(0.0);

    return delta.array().abs().sum() / static_cast<scalar_t>(outputs.rows());
}

scalar_t loss_cauchy_t::eval(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx, tensor3d_map_t Hx)
{
    const auto delta = outputs - targets;

    gx = (2.0 * delta.array() / (1.0 + delta.array().square())).matrix();

    for (tensor_size_t sample = 0, samples = outputs.rows(); sample < samples; ++sample)
    {
        const auto odelta = delta.matrix().row(sample).array();

        Hx.tensor(sample).full(0.0);
        Hx.tensor(sample).diagonal().array() = 2.0 * (1.0 - odelta.square()) / (1.0 + odelta.square()).square();
    }

    return (delta.array().square() + 1.0).log().sum() / static_cast<scalar_t>(outputs.rows());
}

scalar_t loss_hinge_t::eval(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx, tensor3d_map_t Hx)
{
    const auto edges = -outputs.array() * targets.array();

    gx = (-targets.array() * ((1.0 + edges).sign() * 0.5 + 0.5)).matrix();

    Hx.full(0.0);

    return (1.0 + edges).max(0.0).sum() / static_cast<scalar_t>(outputs.rows());
}

scalar_t loss_logistic_t::eval(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx, tensor3d_map_t Hx)
{
    const auto edges = (-outputs.array() * targets.array()).exp();

    gx = ((-targets.array() * edges) / (1.0 + edges)).matrix();

    for (tensor_size_t sample = 0, samples = outputs.rows(); sample < samples; ++sample)
    {
        const auto oedge = edges.matrix().row(sample).array();

        Hx.tensor(sample).full(0.);
        Hx.tensor(sample).diagonal().array() = oedge / (1.0 + oedge).square();
    }

    return (1.0 + edges).log().sum() / static_cast<scalar_t>(outputs.rows());
}
