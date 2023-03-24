#include <nano/linear/result.h>

using namespace nano;
using namespace nano::linear;

fit_result_t::fit_result_t() = default;

fit_result_t::fit_result_t(tensor1d_t bias, tensor2d_t weights, const solver_state_t& state)
    : m_bias(std::move(bias))
    , m_weights(std::move(weights))
    , m_statistics(3)
{
    m_statistics(0) = static_cast<scalar_t>(state.fcalls());
    m_statistics(1) = static_cast<scalar_t>(state.gcalls());
    m_statistics(2) = static_cast<scalar_t>(state.status());
}
