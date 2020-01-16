#pragma once

#include <nano/loss.h>
#include <nano/chrono.h>
#include <nano/solver.h>
#include <nano/iterator.h>

namespace nano
{
    class linear_function_t;

    ///
    /// \brief a linear model is an affine transformation of the flatten input features x:
    ///     y(x) = weights * x + bias.
    ///
    class NANO_PUBLIC linear_model_t
    {
    public:

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
        train_result_t train(const loss_t&, const iterator_t&, const solver_t&);

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
        /// \brief change parameters
        ///
        void normalization(const normalization n) { m_normalization = n; }
        void regularization(const regularization r) { m_regularization = r; }
        void batch(const tensor_size_t batch) { m_batch = batch; }
        void tune_steps(const int steps) { m_tune_steps = steps; }
        void tune_trials(const int trials) { m_tune_trials = trials; }

        ///
        /// \brief access functions
        ///
        auto batch() const { return m_batch.get(); }
        auto tune_steps() const { return m_tune_steps.get(); }
        auto normalization() const { return m_normalization; }
        auto regularization() const { return m_regularization; }
        auto tune_trials() const { return m_tune_trials.get(); }

        const auto& bias() const { return m_bias; }
        const auto& weights() const { return m_weights; }

    private:

        scalar_t train(const linear_function_t&, const solver_t&, scalar_t& best_vd_error);

        // attributes
        tensor2d_t      m_weights;  ///< weight matrix (#inputs, #outputs)
        tensor1d_t      m_bias;     ///< bias vector (#outputs)
        iparam1_t       m_batch{"linear::batch", 1, LE, 32, LE, 4096};          ///< #samples to use at once (= minibatch)
        iparam1_t       m_tune_trials{"linear::tune_trials", 4, LE, 7, LE, 10}; ///<
        iparam1_t       m_tune_steps{"linear::tune_steps", 1, LE, 2, LE, 10};   ///<
        ::nano::regularization  m_regularization{::nano::regularization::none}; ///<
        ::nano::normalization   m_normalization{::nano::normalization::standard};///<
    };
}
