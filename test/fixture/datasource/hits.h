#pragma once

#include <nano/scalar.h>
#include <nano/tensor.h>

namespace nano
{
///
/// \brief return the hit map corresponding to 10% randomly missing feature values (except the target feature).
///
inline auto make_random_hits(const tensor_size_t samples, const tensor_size_t features, const size_t target)
{
    auto hits = make_random_tensor<int8_t>(make_dims(samples, features), 0, 10);

    if (target != string_t::npos)
    {
        hits.matrix().col(static_cast<tensor_size_t>(target)).array() = 1;
    }

    return hits;
}

///
/// \brief return the hit map corresponding to all feature values being given (aka no missing feature value).
///
inline auto make_all_hits(const tensor_size_t samples, const tensor_size_t features)
{
    return make_full_tensor<int8_t>(make_dims(samples, features), 1);
}
} // namespace nano
