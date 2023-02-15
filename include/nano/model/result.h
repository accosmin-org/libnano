#pragma once

#include <any>
#include <nano/string.h>
#include <nano/tensor.h>

namespace nano
{
    ///
    /// \brief statistics collected while fitting a machine learning model for:
    ///     - a set of (train, validation) sample splits (aka folds) and
    ///     - a set of candidate hyper-parameter values (to try/tune).
    ///
    class NANO_PUBLIC fit_result_t
    {
    public:
        enum class split_type
        {
            train,
            valid
        };

        enum class value_type
        {
            errors,
            losses
        };

        struct stats_t
        {
            scalar_t m_mean{0.0};
            scalar_t m_stdev{0.0};
            scalar_t m_count{0.0};
            scalar_t m_per01{0.0};
            scalar_t m_per05{0.0};
            scalar_t m_per10{0.0};
            scalar_t m_per20{0.0};
            scalar_t m_per50{0.0};
            scalar_t m_per80{0.0};
            scalar_t m_per90{0.0};
            scalar_t m_per95{0.0};
            scalar_t m_per99{0.0};
        };

        ///
        /// \brief statistics collected while evaluating a set of hyper-parameter values for all folds.
        ///
        class param_t
        {
        public:
            explicit param_t(tensor1d_t params = tensor1d_t{}, tensor_size_t folds = 0);

            void evaluate(tensor_size_t fold, tensor2d_t train_errors_losses, tensor2d_t valid_errors_losses,
                          std::any extra = std::any{});

            const tensor1d_t& params() const { return m_params; }

            tensor_size_t folds() const { return m_values.size<0>(); }

            stats_t stats(tensor_size_t fold, split_type, value_type) const;

            scalar_t value(split_type = split_type::valid, value_type = value_type::errors) const;

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
        explicit fit_result_t(strings_t param_names = strings_t{});

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
        stats_t stats(value_type) const;

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

    inline bool operator<(const fit_result_t::param_t& lhs, const fit_result_t::param_t& rhs)
    {
        assert(lhs.folds() == rhs.folds());
        return lhs.value() < rhs.value();
    }
} // namespace nano
