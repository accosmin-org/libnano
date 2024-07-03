#pragma once

#include <nano/mlearn/fit_hparam.h>

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
