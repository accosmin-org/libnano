#include <nano/gboost/result.h>
#include <nano/gboost/util.h>
#include <nano/wlearner/util.h>

using namespace nano;
using namespace nano::gboost;

fit_result_t::fit_result_t(const tensor_size_t max_rounds)
    : m_statistics(max_rounds + 1, 7)
{
}

fit_result_t::fit_result_t(fit_result_t&&) = default;

fit_result_t& fit_result_t::operator=(fit_result_t&&) = default;

fit_result_t::fit_result_t(const fit_result_t& other)
    : m_bias(other.m_bias)
    , m_wlearners(wlearner::clone(other.m_wlearners))
    , m_statistics(other.m_statistics)
{
}

fit_result_t& fit_result_t::operator=(const fit_result_t& other)
{
    if (this != &other)
    {
        m_bias       = other.m_bias;
        m_wlearners  = wlearner::clone(other.m_wlearners);
        m_statistics = other.m_statistics;
    }
    return *this;
}

void fit_result_t::update(const tensor_size_t round, const tensor2d_t& errors_values, const indices_t& train_samples,
                          const indices_t& valid_samples, const solver_state_t& state)
{
    m_statistics(round, 0) = mean_error(errors_values, train_samples);
    m_statistics(round, 1) = mean_loss(errors_values, train_samples);
    m_statistics(round, 2) = mean_error(errors_values, valid_samples);
    m_statistics(round, 3) = mean_loss(errors_values, valid_samples);
    m_statistics(round, 4) = static_cast<scalar_t>(state.fcalls);
    m_statistics(round, 5) = static_cast<scalar_t>(state.gcalls);
    m_statistics(round, 6) = static_cast<scalar_t>(state.status);
}

void fit_result_t::update(const tensor_size_t round, const tensor2d_t& errors_values, const indices_t& train_samples,
                          const indices_t& valid_samples, const solver_state_t& state, rwlearner_t&& wlearner)
{
    update(round, errors_values, train_samples, valid_samples, state);
    m_wlearners.emplace_back(std::move(wlearner));
}

void fit_result_t::done(const tensor_size_t optimum_round)
{
    m_wlearners.erase(m_wlearners.begin() + optimum_round, m_wlearners.end());
    m_statistics = m_statistics.slice(0, optimum_round + 1);
    ::nano::wlearner::merge(m_wlearners);
}
