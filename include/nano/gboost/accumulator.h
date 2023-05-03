#pragma once

#include <nano/tensor.h>

namespace nano::gboost
{
///
/// \brief cumulates partial results per thread useful in evaluating gradient boosting-related functions.
///
class NANO_PUBLIC accumulator_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit accumulator_t(tensor_size_t tsize = 1);

    ///
    /// \brief reset the accumulator.
    ///
    void clear();

    ///
    /// \brief update with new values.
    ///
    void update(const tensor1d_cmap_t& values);

    ///
    /// \brief returns the function value and optionally its gradient.
    ///
    scalar_t vgrad(vector_t* gx) const;

    ///
    /// \brief cumulate partial results.
    ///
    accumulator_t& operator+=(const accumulator_t& other);

    ///
    /// \brief normalize the cumulated results with the given number of samples.
    ///
    accumulator_t& operator/=(tensor_size_t samples);

    // attributes
    scalar_t m_vm1{0}; ///< first order momentum of the loss values
    vector_t m_gb1{0}; ///< first order momentum of the gradient wrt scale
};

using accumulators_t = std::vector<accumulator_t>;
} // namespace nano::gboost
