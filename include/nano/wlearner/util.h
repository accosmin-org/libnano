#pragma once

#include <nano/dataset/iterator.h>

namespace nano::wlearner
{
    ///
    /// \brief scale the given tables of coefficients.
    ///
    NANO_PUBLIC void scale(tensor4d_t& tables, const vector_t& scale);

    template <typename toperator>
    void loop_scalar(const dataset_t& dataset, const indices_t& samples, const tensor_size_t feature,
                     const toperator& op)
    {
        select_iterator_t it(dataset, 1U);
        it.loop(samples, feature,
                [&](tensor_size_t, size_t, const scalar_cmap_t& fvalues)
                {
                    for (tensor_size_t i = 0; i < samples.size(); ++i)
                    {
                        if (const auto value = fvalues(i); std::isfinite(value))
                        {
                            op(i, value);
                        }
                    }
                });
    }

    template <typename toperator>
    void loop_sclass(const dataset_t& dataset, const indices_t& samples, const tensor_size_t feature,
                     const toperator& op)
    {
        select_iterator_t it(dataset, 1U);
        it.loop(samples, feature,
                [&](tensor_size_t, size_t, const sclass_cmap_t& fvalues)
                {
                    for (tensor_size_t i = 0; i < samples.size(); ++i)
                    {
                        if (const auto value = fvalues(i); value >= 0)
                        {
                            op(i, value);
                        }
                    }
                });
    }

    template <typename toperator>
    void loop_mclass(const dataset_t& dataset, const indices_t& samples, const tensor_size_t feature,
                     const toperator& op)
    {
        select_iterator_t it(dataset, 1U);
        it.loop(samples, feature,
                [&](tensor_size_t, size_t, const mclass_cmap_t& fvalues)
                {
                    for (tensor_size_t i = 0; i < samples.size(); ++i)
                    {
                        if (const auto values = fvalues.vector(i); values(0) >= 0)
                        {
                            op(i, values);
                        }
                    }
                });
    }
} // namespace nano::wlearner
