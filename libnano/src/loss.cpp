#include <mutex>
#include "loss/hinge.h"
#include "loss/square.h"
#include "loss/cauchy.h"
#include "loss/logistic.h"
#include "loss/classnll.h"
#include "loss/exponential.h"

using namespace nano;

json_t loss_t::config() const
{
    return json_t{};
}

void loss_t::config(const json_t&)
{
}

loss_factory_t& loss_t::all()
{
    static loss_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<square_loss_t> ("square",
            "multivariate regression:     l(y, t) = 1/2 * (y - t)^2");
        manager.add<cauchy_loss_t> ("cauchy",
            "multivariate regression:     l(y, t) = 1/2 * log(1 + (y - t)^2)");

        manager.add<shinge_loss_t>("s-hinge",
            "single-label classification: l(y, t) = max(0, 1 - y*t)");
        manager.add<mhinge_loss_t>("m-hinge",
            "multi-label classification: l(y, t) = max(0, 1 - y*t)");

        manager.add<sclassnll_loss_t>("s-classnll",
            "single-label classification: l(y, t) = log(y.exp().sum()) - log((1 + t).dot(y.exp()))");
        manager.add<mclassnll_loss_t>("m-classnll",
            "multi-label classification: l(y, t) = log(y.exp().sum()) - log((1 + t).dot(y.exp()))");

        manager.add<slogistic_loss_t>("s-logistic",
            "single-label classification: l(y, t) = log(1 + exp(-y*t))");
        manager.add<mlogistic_loss_t>("m-logistic",
            "multi-label classification:  l(y, t) = log(1 + exp(-y*t))");

        manager.add<sexponential_loss_t>("s-exponential",
            "single-label classification: l(y, t) = exp(-y*t)");
        manager.add<mexponential_loss_t>("m-exponential",
            "multi-label classification:  l(y, t) = exp(-y*t)");
    });

    return manager;
}
