#include <nano/core/sampling.h>
#include <solver/gsample/perturbation.h>

using namespace nano;
using namespace nano::gsample;

perturbation_t::perturbation_t(const tensor_size_t n, const scalar_t c)
    : m_zero(vector_t::zero(n))
    , m_ksi(n)
    , m_c(c)
{
}

const vector_t& perturbation_t::generate(const solver_state_t& state, const vector_t& g)
{
    const auto radius = std::max(m_c * state.gx().dot(g) / state.gx().lpNorm<2>(), epsilon0<scalar_t>());
    assert(std::isfinite(radius));
    assert(radius > 0.0);

    auto rng = make_rng();
    sample_from_ball(m_zero, radius, m_ksi, rng);

    return m_ksi;
}
