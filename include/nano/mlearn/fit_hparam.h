#pragma once

#include <nano/mlearn/enums.h>
#include <nano/mlearn/stats.h>
#include <nano/string.h>

namespace nano::ml
{
///
/// \brief statistics collected while evaluating a set of hyper-parameter values for all folds.
///
class NANO_PUBLIC fit_hparam_t
{
public:
    explicit fit_hparam_t(tensor1d_t params = tensor1d_t{}, tensor_size_t folds = 0);

    void evaluate(tensor_size_t fold, tensor2d_t train_errors_losses, tensor2d_t valid_errors_losses);

    const tensor1d_t& params() const { return m_params; }

    const tensor4d_t& values() const { return m_values; }

    tensor_size_t folds() const { return m_values.size<0>(); }

    const string_t& serial_path(tensor_size_t fold) const;
    const string_t& logger_path(tensor_size_t fold) const;

    stats_t stats(tensor_size_t fold, ml::split_type, ml::value_type) const;

    scalar_t value(ml::split_type = ml::split_type::valid, ml::value_type = ml::value_type::errors) const;

    const std::any& extra(tensor_size_t fold) const;

private:
    tensor1d_t m_params;       ///< hyper-parameter values
    tensor4d_t m_values;       ///< evaluation results (fold, train|valid, errors|losses, statistics e.g. mean|stdev)
    strings_t  m_serial_paths; ///< paths to serialized models (fold,)
    strings_t  m_logger_paths; ///< paths to detailed logs (fold,)
};
} // namespace nano::ml
