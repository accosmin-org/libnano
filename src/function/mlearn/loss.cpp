#include <function/mlearn/loss.h>

using namespace nano;

scalar_t loss_mse_t::eval(const linear_model_t& model, matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx,
                          matrix_map_t Hx)
{
    const auto delta = outputs - targets;

    model.eval(gx, Hx, delta);

    return 0.5 * delta.squaredNorm() / static_cast<scalar_t>(outputs.rows());
}

scalar_t loss_mae_t::eval(const linear_model_t& model, matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx,
                          matrix_map_t Hx)
{
    const auto delta = outputs - targets;

    model.eval(gx, Hx, delta.array().sign().matrix());

    return delta.array().abs().sum() / static_cast<scalar_t>(outputs.rows());
}

scalar_t loss_cauchy_t::eval(const linear_model_t& model, matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx,
                             matrix_map_t Hx)
{
    const auto delta = outputs - targets;

    model.eval(gx, Hx, (2.0 * delta.array() / (1.0 + delta.array().square())).matrix());

    return (delta.array().square() + 1.0).log().sum() / static_cast<scalar_t>(outputs.rows());
}

scalar_t loss_hinge_t::eval(const linear_model_t& model, matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx,
                            matrix_map_t Hx)
{
    const auto edges = -outputs.array() * targets.array();

    model.eval(gx, Hx, (-targets.array() * ((1.0 + edges).sign() * 0.5 + 0.5)).matrix());

    return (1.0 + edges).max(0.0).sum() / static_cast<scalar_t>(outputs.rows());
}

scalar_t loss_logistic_t::eval(const linear_model_t& model, matrix_cmap_t outputs, matrix_cmap_t targets,
                               vector_map_t gx, matrix_map_t Hx)
{
    const auto edges = (-outputs.array() * targets.array()).exp();

    model.eval(gx, Hx, ((-targets.array() * edges) / (1.0 + edges)).matrix());

    return (1.0 + edges).log().sum() / static_cast<scalar_t>(outputs.rows());
}
