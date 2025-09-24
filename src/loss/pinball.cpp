#include <nano/loss/pinball.h>

using namespace nano;

pinball_loss_t::pinball_loss_t()
    : loss_t("pinball")
{
    convex(true);
    smooth(false);

    register_parameter(parameter_t::make_scalar("loss::pinball::alpha", 0.0, LE, 0.5, LE, 1.0));
}

rloss_t pinball_loss_t::clone() const
{
    return std::make_unique<pinball_loss_t>(*this);
}

void pinball_loss_t::error(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t errors) const
{
    value(targets, outputs, errors);
}

void pinball_loss_t::value(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t values) const
{
    assert(targets.dims() == outputs.dims());
    assert(values.size() == targets.size<0>());

    const auto alpha = parameter("loss::pinball::alpha").value<scalar_t>();

    for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++i)
    {
        const auto itarget = targets.array(i);
        const auto ioutput = outputs.array(i);

        values(i) = (alpha * (itarget - ioutput).max(0.0) + (1.0 - alpha) * (ioutput - itarget).max(0.0)).sum();
    }
}

void pinball_loss_t::vgrad(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor4d_map_t vgrads) const
{
    assert(targets.dims() == vgrads.dims());
    assert(targets.dims() == outputs.dims());

    const auto alpha = parameter("loss::pinball::alpha").value<scalar_t>();

    for (tensor_size_t i = 0, samples = targets.size<0>(); i < samples; ++i)
    {
        const auto itarget = targets.array(i);
        const auto ioutput = outputs.array(i);

        vgrads.array(i) = -alpha + 0.5 * (1.0 - (itarget - ioutput).sign());
    }
}

void pinball_loss_t::vhess([[maybe_unused]] tensor4d_cmap_t targets, [[maybe_unused]] tensor4d_cmap_t outputs,
                           [[maybe_unused]] tensor3d_map_t vhesss) const
{
    assert(targets.size<0>() == vhesss.size<0>());
    assert(targets.size<1>() * targets.size<2>() * targets.size<3>() == vhesss.size<1>());
    assert(targets.size<1>() * targets.size<2>() * targets.size<3>() == vhesss.size<2>());
    assert(targets.dims() == outputs.dims());
    assert(false);
}
