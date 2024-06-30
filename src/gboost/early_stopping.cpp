#include <nano/gboost/early_stopping.h>
#include <nano/gboost/util.h>

using namespace nano;
using namespace nano::gboost;

early_stopping_t::early_stopping_t(tensor2d_t values)
    : m_value(std::numeric_limits<scalar_t>::max())
    , m_values(std::move(values))
{
}

bool early_stopping_t::done(const tensor2d_t& errors_losses, const indices_t& train_samples,
                            const indices_t& valid_samples, const rwlearners_t& wlearners, const scalar_t epsilon,
                            const size_t patience)
{
    const auto train_value = mean_error(errors_losses, train_samples);
    const auto valid_value = mean_error(errors_losses, valid_samples);

    // training error is too small, stop
    if (train_value < epsilon)
    {
        m_value  = valid_value;
        m_round  = wlearners.size();
        m_values = errors_losses;
        return true;
    }

    // significant improvement, continue
    // OR refitting step when no validation samples are given, so don't stop until the optimum number of boosting rounds
    else if (valid_value < m_value - epsilon || valid_samples.size() == 0)
    {
        m_value  = valid_value;
        m_round  = wlearners.size();
        m_values = errors_losses;
        return false;
    }

    // no significant improvement, but can wait a bit more
    else if (wlearners.size() < m_round + patience)
    {
        return false;
    }

    // no significant improvement in awhile, stop
    else
    {
        return true;
    }
}
