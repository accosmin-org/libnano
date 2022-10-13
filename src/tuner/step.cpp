#include <nano/tuner/step.h>

using namespace nano;

tuner_step_t::tuner_step_t() = default;

tuner_step_t::tuner_step_t(indices_t igrid, tensor1d_t param, scalar_t value)
    : m_igrid(std::move(igrid))
    , m_param(std::move(param))
    , m_value(value)
{
}
