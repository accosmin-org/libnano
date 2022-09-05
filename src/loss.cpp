#include <mutex>
#include <nano/loss/flatten.h>

using namespace nano;

loss_t::loss_t(string_t id)
    : clonable_t(std::move(id))
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
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
