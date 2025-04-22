#pragma once

#include <nano/tensor.h>

namespace nano
{
template <class tvectoru, class tvectordu>
scalar_t make_umax(const tvectoru& u, const tvectordu& du)
{
    assert(u.size() == du.size());

    auto step = std::numeric_limits<scalar_t>::max();
    for (tensor_size_t i = 0, size = u.size(); i < size; ++i)
    {
        if (du(i) < 0.0)
        {
            step = std::min(step, -u(i) / du(i));
        }
    }

    return std::min(step, 1.0);
}

scalar_t make_xmax(const vector_t& x, const vector_t& dx, const matrix_t& G, const vector_t& h)
{
    assert(x.size() == dx.size());
    assert(x.size() == G.cols());
    assert(h.size() == G.rows());

    return make_umax(h - G * x, -G * dx);
}
}
