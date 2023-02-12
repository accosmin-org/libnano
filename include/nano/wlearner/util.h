#pragma once

#include <nano/dataset/iterator.h>
#include <nano/wlearner.h>

namespace nano::wlearner
{
    ///
    /// \brief scale the given tables of coefficients.
    ///
    NANO_PUBLIC void scale(tensor4d_t& tables, const vector_t& scale);

    ///
    /// \brief merge in-place if possible the given weak learners.
    ///
    NANO_PUBLIC void merge(rwlearners_t&);

    ///
    /// \brief clone the given weak learners.
    ///
    NANO_PUBLIC rwlearners_t clone(const rwlearners_t&);

    ///
    /// \brief loop over the feature values of the given scalar feature and samples.
    ///
    template <typename toperator>
    void loop_scalar(const dataset_t& dataset, const indices_t& samples, const tensor_size_t feature,
                     const toperator& op)
    {
        const auto iterator = select_iterator_t{dataset};
        iterator.loop(samples, feature,
                      [&](tensor_size_t, size_t, const scalar_cmap_t& fvalues)
                      {
                          for (tensor_size_t i = 0; i < samples.size(); ++i)
                          {
                              if (const auto value = fvalues(i); std::isfinite(value))
                              {
                                  op(i, fvalues(i));
                              }
                          }
                      });
    }

    ///
    /// \brief loop over the feature values of the single-label scalar feature and samples.
    ///
    template <typename toperator>
    void loop_sclass(const dataset_t& dataset, const indices_t& samples, const tensor_size_t feature,
                     const toperator& op)
    {
        const auto iterator = select_iterator_t{dataset};
        iterator.loop(samples, feature,
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

    ///
    /// \brief loop over the feature values of the multi-label scalar feature and samples.
    ///
    template <typename toperator>
    void loop_mclass(const dataset_t& dataset, const indices_t& samples, const tensor_size_t feature,
                     const toperator& op)
    {
        const auto iterator = select_iterator_t{dataset};
        iterator.loop(samples, feature,
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
