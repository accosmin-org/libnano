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

void loss_mse_t::hx(matrix_cmap_t outputs, matrix_cmap_t, tensor3d_map_t hx)
{
    const auto [samples, osize] = outputs.dims();

    for (tensor_size_t sample = 0; sample < samples; ++sample)
    {
        hx.tensor(sample) = matrix_t::identity(osize, osize);
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

void loss_mae_t::hx(matrix_cmap_t, matrix_cmap_t, tensor3d_map_t hx)
{
    hx.full(0.0);
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

void loss_cauchy_t::hx(matrix_cmap_t outputs, matrix_cmap_t targets, tensor3d_map_t hx)
{
    const auto samples = outputs.rows();

    for (tensor_size_t sample = 0; sample < samples; ++sample)
    {
        const auto odelta = outputs.array(sample) - targets.array(sample);

        hx.tensor(sample) = (2.0 * (1.0 - odelta.square()) / (1.0 + odelta.square()).square()).matrix().asDiagonal();
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

void loss_hinge_t::hx(matrix_cmap_t, matrix_cmap_t, tensor3d_map_t hx)
{
    hx.full(0.0);
}

scalar_t loss_logistic_t::fx(matrix_cmap_t outputs, matrix_cmap_t targets)
{
    const auto samples = outputs.rows();
    // const auto edges   = (-outputs.array() * targets.array()).exp();

    // return edges.log1p().sum() / static_cast<scalar_t>(samples);

    auto fx = 0.0;
    for (tensor_size_t sample = 0; sample < samples; ++sample)
    {
        const auto output = outputs.array(sample);
        const auto target = targets.array(sample);

        for (tensor_size_t i = 0, size = output.size(); i < size; ++i)
        {
            const auto x = -output(i) * target(i);
            fx += (x < 1.0) ? std::log1p(std::exp(x)) : (x + std::log1p(std::exp(-x)));
        }
    }

    return fx / static_cast<scalar_t>(samples);
}

void loss_logistic_t::gx(matrix_cmap_t outputs, matrix_cmap_t targets, matrix_map_t gx)
{
    const auto edges = (-outputs.array() * targets.array()).exp();

    gx = ((-targets.array() * edges) / (1.0 + edges)).matrix();
}

void loss_logistic_t::hx(matrix_cmap_t outputs, matrix_cmap_t targets, tensor3d_map_t hx)
{
    const auto samples = outputs.rows();

    for (tensor_size_t sample = 0; sample < samples; ++sample)
    {
        // const auto oedge = (-outputs.array(sample) * targets.array(sample)).exp();
        //
        // hx.matrix(sample) = (oedge / (1.0 + oedge).square()).matrix().asDiagonal();

        const auto output = outputs.array(sample);
        const auto target = targets.array(sample);

        hx.array(sample) = 0.0;
        for (tensor_size_t i = 0, size = output.size(); i < size; ++i)
        {
            const auto x = output(i) * target(i);
            const auto h =
                (x < 1.0) ? (std::exp(x) / square(1.0 + std::exp(x))) : (std::exp(-x) / square(1.0 + std::exp(-x)));

            hx(sample, i, i) = target(i) * target(i) * h;
        }
    }
}
