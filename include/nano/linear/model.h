#pragma once

#include <nano/loss.h>
#include <nano/chrono.h>
#include <nano/solver.h>
#include <nano/iterator.h>

namespace nano
{
    ///
    /// \brief a linear model is an affine transformation of the flatten input features x:
    ///     y(x) = weights * x + bias.
    ///
    class NANO_PUBLIC linear_model_t
    {
    public:

        ///
        /// \brief regularization methods.
        ///
        enum class regularization
        {
            none,           ///< no regularization
            lasso,          ///< like in LASSO
            ridge,          ///< like in ridge regression, weight decay or Tikhonov regularization
            elastic,        ///< like in elastic net regularization
            variance        ///< like in VadaBoost
        };

        ///
        /// \brief summarizes training for a fold:
        ///     - [train,valid,test] x [loss, error, loss of the averaged model, error of the averaged model]
        ///     - optimum factor for the L1-norm regularization (lasso, elastic)
        ///     - optimum factor for the L2-norm regularization (ridge, elastic)
        ///     - optimum factor for the variance regularization (variance)
        ///     - training and evaluation time in milliseconds
        ///
        struct train_fold_t
        {
            scalar_t        m_tr_loss{0}, m_tr_error{0}, m_avg_tr_loss{0}, m_avg_tr_error{0};   ///<
            scalar_t        m_vd_loss{0}, m_vd_error{0}, m_avg_vd_loss{0}, m_avg_vd_error{0};   ///<
            scalar_t        m_te_loss{0}, m_te_error{0}, m_avg_te_loss{0}, m_avg_te_error{0};   ///<
            scalar_t        m_l1reg{0}, m_l2reg{0}, m_vAreg{0};                                 ///<
            milliseconds_t  m_train_time{0}, m_eval_time{0};                                    ///<
        };

        using train_result_t = std::vector<train_fold_t>;

        ///
        /// \brief default constructor
        ///
        linear_model_t() = default;

        ///
        /// \brief train the linear model on the given samples.
        ///
        train_result_t train(const loss_t&, const iterator_t&, const solver_t&, regularization,
            int batch = 32, int max_trials_per_tune_step = 7, int tune_steps = 2);

        ///
        /// \brief save the trained model to disk
        ///
        void save(const string_t& filepath) const;

        ///
        /// \brief load the trained model from disk
        ///
        void load(const string_t& filepath);

        ///
        /// \brief compute the predictions on the given inputs/samples
        ///
        void predict(const tensor4d_cmap_t& inputs, tensor4d_t& outputs) const;
        void predict(const tensor4d_cmap_t& inputs, tensor4d_map_t&& outputs) const;

        ///
        /// \brief compute the predictions for all samples in the given fold
        ///
        void predict(const iterator_t& iterator, const fold_t&, tensor4d_t& outputs) const;
        void predict(const iterator_t& iterator, const fold_t&, tensor4d_map_t&& outputs) const;

        ///
        /// \brief evaluate the loss error for all samples in the given fold
        ///
        static void evaluate(const iterator_t& iterator, const fold_t& fold, tensor_size_t batch,
            const loss_t& loss, const tensor2d_cmap_t& weights, const tensor1d_cmap_t& bias,
            tensor1d_map_t&& errors);

        ///
        /// \brief evaluate the loss value and error for all samples in the given fold
        ///
        static void evaluate(const iterator_t& iterator, const fold_t& fold, tensor_size_t batch,
            const loss_t& loss, const tensor2d_cmap_t& weights, const tensor1d_cmap_t& bias,
            tensor1d_map_t&& values, tensor1d_map_t&& errors);

        ///
        /// \brief access parameters
        ///
        const auto& bias() const { return m_bias; }
        const auto& weights() const { return m_weights; }

    private:

        scalar_t train(
            const loss_t&, const iterator_t&, size_t fold, const solver_t&, int batch,
            scalar_t l1reg, scalar_t l2reg, scalar_t vAreg, scalar_t& best_vd_error);

        // attributes
        tensor2d_t          m_weights;  ///< weight matrix (#inputs, #outputs)
        tensor1d_t          m_bias;     ///< bias vector (#outputs)
    };

    template <>
    inline enum_map_t<linear_model_t::regularization> enum_string<linear_model_t::regularization>()
    {
        return
        {
            { linear_model_t::regularization::none,     "none" },
            { linear_model_t::regularization::lasso,    "lasso" },
            { linear_model_t::regularization::ridge,    "ridge" },
            { linear_model_t::regularization::elastic,  "elastic" },
            { linear_model_t::regularization::variance, "variance" }
        };
    }
}
