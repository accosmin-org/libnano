#include <mutex>
#include "losses/hinge.h"
#include "losses/square.h"
#include "losses/cauchy.h"
#include "losses/logistic.h"
#include "losses/classnll.h"
#include "losses/exponential.h"

using namespace nano;

void loss_t::to_json(json_t&) const
{
}

void loss_t::from_json(const json_t&)
{
}

loss_factory_t& nano::get_losses()
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
