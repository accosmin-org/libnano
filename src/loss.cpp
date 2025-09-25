#include <mutex>
#include <nano/critical.h>
#include <nano/loss/flatten.h>
#include <nano/loss/pinball.h>

using namespace nano;

namespace
{
template <class tdims>
void check_compatible(const char* const name1, const tdims dims1, const char* const name2, const tdims dims2)
{
    critical(dims1 == dims2, "loss: incompatible dimension-wise ", name1, " and ", name2, " (", dims1, ") vs. (", dims2,
             ")!");
}
} // namespace

loss_t::loss_t(string_t id)
    : typed_t(std::move(id))
{
}

void loss_t::convex(bool convex)
{
    m_convex = convex;
}

void loss_t::smooth(bool smooth)
{
    m_smooth = smooth;
}

tensor7d_dims_t loss_t::make_hess_dims(const tensor4d_cmap_t targets)
{
    return make_hess_dims(targets.dims());
}

tensor7d_dims_t loss_t::make_hess_dims(const tensor4d_dims_t targets_dims)
{
    return make_dims(targets_dims[0], targets_dims[1], targets_dims[2], targets_dims[3], targets_dims[1],
                     targets_dims[2], targets_dims[3]);
}

tensor7d_dims_t loss_t::make_hess_dims(const tensor_size_t samples, const tensor3d_dims_t target_dims)
{
    return make_hess_dims(cat_dims(samples, target_dims));
}

void loss_t::error(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t errors) const
{
    check_compatible("outputs", outputs.dims(), "targets", targets.dims());
    check_compatible("error buffer", errors.dims(), "samples", make_dims(targets.size<0>()));

    do_error(targets, outputs, errors);
}

void loss_t::value(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t values) const
{
    check_compatible("outputs", outputs.dims(), "targets", targets.dims());
    check_compatible("value buffer", values.dims(), "samples", make_dims(targets.size<0>()));

    do_value(targets, outputs, values);
}

void loss_t::vgrad(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor4d_map_t vgrads) const
{
    check_compatible("outputs", outputs.dims(), "targets", targets.dims());
    check_compatible("gradient buffer", vgrads.dims(), "targets", targets.dims());

    do_vgrad(targets, outputs, vgrads);
}

void loss_t::vhess(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor7d_map_t vhesss) const
{
    check_compatible("outputs", outputs.dims(), "targets", targets.dims());
    check_compatible("hessian buffer", vhesss.dims(), "cross-targets", make_hess_dims(targets.dims()));

    do_vhess(targets, outputs, vhesss);
}

void loss_t::error(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_t& errors) const
{
    errors.resize(targets.size<0>());
    error(targets, outputs, errors.tensor());
}

void loss_t::value(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_t& values) const
{
    values.resize(targets.size<0>());
    value(targets, outputs, values.tensor());
}

void loss_t::vgrad(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor4d_t& vgrads) const
{
    vgrads.resize(targets.dims());
    vgrad(targets, outputs, vgrads.tensor());
}

void loss_t::vhess(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor7d_t& vhesss) const
{
    vhesss.resize(make_hess_dims(targets));
    vhess(targets, outputs, vhesss.tensor());
}

factory_t<loss_t>& loss_t::all()
{
    static auto manager = factory_t<loss_t>{};
    const auto  op      = []()
    {
        manager.add<mae_loss_t>("(mean) absolute error (multivariate regression)");
        manager.add<mse_loss_t>("(mean) squared error (multivariate regression)");
        manager.add<cauchy_loss_t>("cauchy loss (multivariate regression)");

        manager.add<mhinge_loss_t>("hinge loss (multi-label classification)");
        manager.add<shinge_loss_t>("hinge loss (single-label classification)");

        manager.add<msquared_hinge_loss_t>("squared hinge loss (multi-label classification)");
        manager.add<ssquared_hinge_loss_t>("squared hinge loss (single-label classification)");

        manager.add<sclassnll_loss_t>("class negative log likehoold (single-label classification)");

        manager.add<msavage_loss_t>("savage loss (multi-label classification)");
        manager.add<ssavage_loss_t>("savage loss (single-label classification)");

        manager.add<mtangent_loss_t>("tangent loss (multi-label classification)");
        manager.add<stangent_loss_t>("tangent loss (single-label classification)");

        manager.add<mlogistic_loss_t>("logistic loss (multi-label classification)");
        manager.add<slogistic_loss_t>("logistic loss (single-label classification)");

        manager.add<sexponential_loss_t>("exponential loss (single-label classification)");
        manager.add<mexponential_loss_t>("exponential loss (multi-label classification)");

        manager.add<pinball_loss_t>("pinball loss (quantile regression)");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
