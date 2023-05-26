#include <nano/gboost/result.h>
#include <nano/gboost/util.h>
#include <nano/wlearner/util.h>

using namespace nano;
using namespace nano::gboost;

result_t::result_t(const tensor2d_t* errors_values, const indices_t* train_samples, const indices_t* valid_samples,
                   const tensor_size_t max_rounds)
    : m_errors_values(errors_values)
    , m_train_samples(train_samples)
    , m_valid_samples(valid_samples)
    , m_statistics(max_rounds + 1, 8)
{
}

result_t::result_t(result_t&&) noexcept = default;

result_t& result_t::operator=(result_t&&) noexcept = default;

result_t::result_t(const result_t& other)
    : m_errors_values(other.m_errors_values)
    , m_train_samples(other.m_train_samples)
    , m_valid_samples(other.m_valid_samples)
    , m_bias(other.m_bias)
    , m_wlearners(wlearner::clone(other.m_wlearners))
    , m_statistics(other.m_statistics)
{
}

result_t& result_t::operator=(const result_t& other)
{
    if (this != &other)
    {
        m_errors_values = other.m_errors_values;
        m_train_samples = other.m_train_samples;
        m_valid_samples = other.m_valid_samples;
        m_bias          = other.m_bias;
        m_wlearners     = wlearner::clone(other.m_wlearners);
        m_statistics    = other.m_statistics;
    }
    return *this;
}

result_t::~result_t() = default;

void result_t::update(const tensor_size_t round, const scalar_t shrinkage_ratio, const solver_state_t& state)
{
    assert(m_errors_values != nullptr);
    assert(m_train_samples != nullptr);
    assert(m_valid_samples != nullptr);

    m_statistics(round, 0) = mean_error(*m_errors_values, *m_train_samples);
    m_statistics(round, 1) = mean_loss(*m_errors_values, *m_train_samples);
    m_statistics(round, 2) = mean_error(*m_errors_values, *m_valid_samples);
    m_statistics(round, 3) = mean_loss(*m_errors_values, *m_valid_samples);
    m_statistics(round, 4) = shrinkage_ratio;
    m_statistics(round, 5) = static_cast<scalar_t>(state.fcalls());
    m_statistics(round, 6) = static_cast<scalar_t>(state.gcalls());
    m_statistics(round, 7) = static_cast<scalar_t>(state.status());
}

void result_t::update(const tensor_size_t round, const scalar_t shrinkage_ratio, const solver_state_t& state,
                      rwlearner_t&& wlearner)
{
    update(round, shrinkage_ratio, state);
    m_wlearners.emplace_back(std::move(wlearner));
}

void result_t::done(const tensor_size_t optimum_round)
{
    m_wlearners.erase(m_wlearners.begin() + optimum_round, m_wlearners.end());
    m_statistics = m_statistics.slice(0, optimum_round + 1);
    ::nano::wlearner::merge(m_wlearners);
}
