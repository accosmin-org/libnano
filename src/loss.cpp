#include <mutex>
#include <nano/loss/flatten.h>

using namespace nano;

loss_factory_t& loss_t::all()
{
    static loss_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<cauchy_loss_t>("cauchy", "cauchy loss (multivariate regression)");
        manager.add<squared_loss_t>("squared", "squared error (multivariate regression)");
        manager.add<absolute_loss_t>("absolute", "absolute error (multivariate regression)");

        manager.add<mhinge_loss_t>("m-hinge", "hinge loss (multi-label classification)");
        manager.add<shinge_loss_t>("s-hinge", "hinge loss (single-label classification)");

        manager.add<msquared_hinge_loss_t>("m-squared-hinge", "squared hinge loss (multi-label classification)");
        manager.add<ssquared_hinge_loss_t>("s-squared-hinge", "squared hinge loss (single-label classification)");

        manager.add<sclassnll_loss_t>("s-classnll", "class negative log likehoold (single-label classification)");

        manager.add<msavage_loss_t>("m-savage", "savage loss (multi-label classification)");
        manager.add<ssavage_loss_t>("s-savage", "savage loss (single-label classification)");

        manager.add<mtangent_loss_t>("m-tangent", "tangent loss (multi-label classification)");
        manager.add<stangent_loss_t>("s-tangent", "tangent loss (single-label classification)");

        manager.add<mlogistic_loss_t>("m-logistic", "logistic loss (multi-label classification)");
        manager.add<slogistic_loss_t>("s-logistic", "logistic loss (single-label classification)");

        manager.add<sexponential_loss_t>("s-exponential", "exponential loss (single-label classification)");
        manager.add<mexponential_loss_t>("m-exponential", "exponential loss (multi-label classification)");
    });

    return manager;
}
