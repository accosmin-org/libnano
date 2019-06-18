#pragma once

#include <algorithm>

namespace nano
{
    ///
    /// \brief set the elements of the given tensors to random values.
    ///
    template <typename tdistribution, typename trng, typename ttensor>
    void set_random(tdistribution&& distribution, trng&& rgen, ttensor&& tensor)
    {
        std::for_each(tensor.data(), tensor.data() + tensor.size(), [&] (auto& v) { v = distribution(rgen); });
    }

    template <typename tdistribution, typename trng, typename ttensor, typename... tothers>
    void set_random(tdistribution&& distribution, trng&& rgen, ttensor&& tensor, tothers&&... others)
    {
        set_random(distribution, rgen, tensor);
        set_random(distribution, rgen, others...);
    }

    ///
    /// \brief add random values to the elements of the given tensors.
    ///
    template <typename tdistribution, typename trng, typename ttensor>
    void add_random(tdistribution&& distribution, trng&& rgen, ttensor&& tensor)
    {
        std::for_each(tensor.data(), tensor.data() + tensor.size(), [&] (auto& v) { v += distribution(rgen); });
    }

    template <typename tdistribution, typename trng, typename ttensor, typename... tothers>
    void add_random(tdistribution&& distribution, trng&& rgen, ttensor&& tensor, tothers&&... others)
    {
        add_random(distribution, rgen, tensor);
        add_random(distribution, rgen, others...);
    }
}
