#pragma once

#include <nano/solver/state.h>

namespace nano::gsample
{
class perturbation_t
{
public:
    perturbation_t(tensor_size_t n, scalar_t c);

    const vector_t& generate(const solver_state_t& state, const vector_t& g);

private:
    // attributes
    vector_t m_zero;
    vector_t m_ksi;
    scalar_t m_c{1e-6};
};
} // namespace nano::gsample
