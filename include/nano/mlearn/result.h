#pragma once

#include <any>
#include <nano/mlearn/enums.h>
#include <nano/mlearn/stats.h>
#include <nano/string.h>

namespace nano::ml
{
///
/// \brief statistics collected while fitting a machine learning model for:
///     - a set of (train, validation) sample splits (aka folds) and
///     - a set of candidate hyper-parameter values to tune.
///
class NANO_PUBLIC result_t
{
public:
    ///
    /// \brief statistics collected while evaluating a set of hyper-parameter values for all folds.
    ///
    class NANO_PUBLIC param_t
    {
    public:
        explicit param_t(tensor1d_t params = tensor1d_t{}, tensor_size_t folds = 0);

        void evaluate(tensor_size_t fold, tensor2d_t train_errors_losses, tensor2d_t valid_errors_losses,
                      std::any extra = std::any{});

        const tensor1d_t& params() const { return m_params; }

        const tensor4d_t& values() const { return m_values; }

        tensor_size_t folds() const { return m_values.size<0>(); }

        stats_t stats(tensor_size_t fold, ml::split_type, ml::value_type) const;

        scalar_t value(ml::split_type = ml::split_type::valid, ml::value_type = ml::value_type::errors) const;

        const std::any& extra(tensor_size_t fold) const;

    private:
        using anys_t = std::vector<std::any>;

        tensor1d_t m_params; ///< hyper-parameter values
        tensor4d_t m_values; ///< evaluation (fold, train|valid, errors|losses, statistics e.g. mean|stdev)
        anys_t     m_extras; ///< model specific data per fold
    };

    using params_t = std::vector<param_t>;

    ///
    /// \brief constructor
    ///
    explicit result_t(strings_t param_names = strings_t{});

    ///
    /// \brief add the evaluation results of a hyper-parameter trial.
    ///
    void add(param_t);

    ///
    /// \brief return the optimum hyper-parameters from all stored trials.
    ///
    const param_t& optimum() const;

    ///
    /// \brief set the evaluation results for the optimum hyper-parameters.
    ///
    void evaluate(tensor2d_t errors_losses);

    ///
    /// \brief returns the hyper-parameter names.
    ///
    const strings_t& param_names() const { return m_param_names; }

    ///
    /// \brief returns the set of hyper-parameters that have been evaluated.
    ///
    const params_t& param_results() const { return m_param_results; }

    ///
    /// \brief returns the statistics associated to the optimum hyper-parameters.
    ///
    stats_t stats(ml::value_type) const;

    ///
    /// \brief returns the closest parameter to the given one.
    ///
    const param_t* closest(const tensor1d_cmap_t& params) const;

private:
    // attributes
    strings_t  m_param_names;   ///< name of the hyper-parameters
    params_t   m_param_results; ///< results obtained by evaluating candidate hyper-parameters
    tensor2d_t m_optim_values;  ///< optimum's evaluation (errors|losses, statistics e.g. mean|stdev)
};

inline bool operator<(const result_t::param_t& lhs, const result_t::param_t& rhs)
{
    assert(lhs.folds() == rhs.folds());
    return lhs.value() < rhs.value();
}
} // namespace nano::ml
