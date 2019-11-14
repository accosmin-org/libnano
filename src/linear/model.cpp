#include <fstream>
#include <nano/tune.h>
#include <nano/logger.h>
#include <nano/version.h>
#include <nano/linear/util.h>
#include <nano/linear/model.h>
#include <nano/tensor/stream.h>
#include <nano/linear/function.h>

using namespace nano;

linear_model_t::train_result_t linear_model_t::train(
    const loss_t& loss, const iterator_t& iterator, const solver_t& solver,
    const regularization reg, const int batch, const int max_trials_per_tune_step, const int tune_steps)
{
    log_info() << "training linear model...";

    tensor1d_t avg_bias(nano::size(iterator.tdim()));
    tensor2d_t avg_weights(nano::size(iterator.idim()), nano::size(iterator.tdim()));

    avg_bias.zero();
    avg_weights.zero();

    // train a model for each fold ...
    train_result_t results{iterator.folds()};
    for (size_t fold = 0, folds = iterator.folds(); fold < folds; ++ fold)
    {
        const auto tr_fold = fold_t{fold, protocol::train};
        const auto vd_fold = fold_t{fold, protocol::valid};
        const auto te_fold = fold_t{fold, protocol::test};

        tensor1d_t tr_values(iterator.samples(tr_fold)), tr_errors(tr_values.size());
        tensor1d_t vd_values(iterator.samples(vd_fold)), vd_errors(vd_values.size());
        tensor1d_t te_values(iterator.samples(te_fold)), te_errors(te_values.size());

        // tune the regularization factors (if any)
        const timer_t start_train;

        scalar_t best_l1reg = 0.0;
        scalar_t best_l2reg = 0.0;
        scalar_t best_vAreg = 0.0;
        scalar_t best_vd_error = std::numeric_limits<scalar_t>::max();

        switch (reg)
        {
        case regularization::lasso:
            std::tie(best_vd_error, best_l1reg) = nano::tune(
                nano::pow10_space_t{-6.0, +6.0},
                [&] (const scalar_t l1reg)
                {
                    return train(loss, iterator, fold, solver, batch, l1reg, 0.0, 0.0, best_vd_error);
                },
                max_trials_per_tune_step, tune_steps);
            break;

        case regularization::ridge:
            std::tie(best_vd_error, best_l2reg) = nano::tune(
                nano::pow10_space_t{-6.0, +6.0},
                [&] (const scalar_t l2reg)
                {
                    return train(loss, iterator, fold, solver, batch, 0.0, l2reg, 0.0, best_vd_error);
                },
                max_trials_per_tune_step, tune_steps);
            break;

        case regularization::variance:
            std::tie(best_vd_error, best_vAreg) = nano::tune(
                nano::pow10_space_t{-6.0, +6.0},
                [&] (const scalar_t vAreg)
                {
                    return train(loss, iterator, fold, solver, batch, 0.0, 0.0, vAreg, best_vd_error);
                },
                max_trials_per_tune_step, tune_steps);
            break;

        case regularization::elastic:
            std::tie(best_vd_error, best_l1reg, best_l2reg) = nano::tune(
                nano::pow10_space_t{-8.0, +6.0},
                nano::pow10_space_t{-8.0, +6.0},
                [&] (const scalar_t l1reg, const scalar_t l2reg)
                {
                    return train(loss, iterator, fold, solver, batch, l1reg, l2reg, 0.0, best_vd_error);
                },
                max_trials_per_tune_step, tune_steps);
            break;

        case regularization::none:
            train(loss, iterator, fold, solver, batch, 0.0, 0.0, 0.0, best_vd_error);
            break;

        default:
            critical(true, "unhandled regularization method when training linear models");
        }

        auto& result = results[fold];
        result.m_l1reg = best_l1reg;
        result.m_l2reg = best_l2reg;
        result.m_vAreg = best_vAreg;
        result.m_train_time = start_train.milliseconds();

        // update the average model
        avg_bias.vector() = (avg_bias.vector() * fold + m_bias.vector()) / (fold + 1);
        avg_weights.vector() = (avg_weights.vector() * fold + m_weights.vector()) / (fold + 1);

        // evaluate the current tuned model
        const timer_t start_eval;
        evaluate(iterator, tr_fold, batch, loss, m_weights, m_bias, tr_values.tensor(), tr_errors.tensor());
        evaluate(iterator, vd_fold, batch, loss, m_weights, m_bias, vd_values.tensor(), vd_errors.tensor());
        evaluate(iterator, te_fold, batch, loss, m_weights, m_bias, te_values.tensor(), te_errors.tensor());

        result.m_tr_loss = tr_values.vector().mean(), result.m_tr_error = tr_errors.vector().mean();
        result.m_vd_loss = vd_values.vector().mean(), result.m_vd_error = vd_errors.vector().mean();
        result.m_te_loss = te_values.vector().mean(), result.m_te_error = te_errors.vector().mean();

        assert(std::fabs(best_vd_error - result.m_vd_error) < 1e-8);

        // evaluate the averaged model
        evaluate(iterator, tr_fold, batch, loss, avg_weights, avg_bias, tr_values.tensor(), tr_errors.tensor());
        evaluate(iterator, vd_fold, batch, loss, avg_weights, avg_bias, vd_values.tensor(), vd_errors.tensor());
        evaluate(iterator, te_fold, batch, loss, avg_weights, avg_bias, te_values.tensor(), te_errors.tensor());

        result.m_avg_tr_loss = tr_values.vector().mean(), result.m_avg_tr_error = tr_errors.vector().mean();
        result.m_avg_vd_loss = vd_values.vector().mean(), result.m_avg_vd_error = vd_errors.vector().mean();
        result.m_avg_te_loss = te_values.vector().mean(), result.m_avg_te_error = te_errors.vector().mean();
        result.m_eval_time = start_eval.milliseconds();

        //
        log_info() << std::setprecision(8) << std::fixed << ">>> current"
            << ":tr=" << result.m_tr_loss << "|" << result.m_tr_error
            << ",vd=" << result.m_vd_loss << "|" << result.m_vd_error
            << ",te=" << result.m_te_loss << "|" << result.m_te_error << ".";

        log_info() << std::setprecision(8) << std::fixed << ">>> average"
            << ":tr=" << result.m_avg_tr_loss << "|" << result.m_avg_tr_error
            << ",vd=" << result.m_avg_vd_loss << "|" << result.m_avg_vd_error
            << ",te=" << result.m_avg_te_loss << "|" << result.m_avg_te_error << ".";

        log_info() << std::setprecision(8) << std::fixed << ">>> optimum"
            << ":l1reg=" << result.m_l1reg << ",l2reg=" << result.m_l2reg << ",vAreg=" << result.m_vAreg << ".";
    }

    // the final model is the average across all folds
    m_bias = avg_bias;
    m_weights = avg_weights;

    // OK
    return results;
}

scalar_t linear_model_t::train(
    const loss_t& loss, const iterator_t& iterator, const size_t fold, const solver_t& solver, const int batch,
    const scalar_t l1reg, const scalar_t l2reg, const scalar_t vAreg,
    scalar_t& best_vd_error)
{
    auto function = linear_function_t{loss, iterator, fold_t{fold, protocol::train}};
    function.batch(batch);
    function.l1reg(l1reg);
    function.l2reg(l2reg);
    function.vAreg(vAreg);

    const auto state = solver.minimize(function, vector_t::Zero(function.size()));

    const auto bias = function.bias(state.x);
    const auto weights = function.weights(state.x);
    tensor1d_t vd_errors(iterator.samples(fold_t{fold, protocol::valid}));
    evaluate(iterator, fold_t{fold, protocol::valid}, batch, loss, weights, bias, vd_errors.tensor());

    const auto vd_error = vd_errors.vector().mean();
    const auto better = vd_error < best_vd_error;

    log_info() << std::setprecision(8) << std::fixed
        << "fold=" << (fold + 1) << "|" << iterator.folds()
        << ":l1reg=" << l1reg << ",l2reg=" << l2reg << ",vAreg=" << vAreg
        << ",iters=" << state.m_iterations << ",calls=" << state.m_fcalls << "/" << state.m_gcalls
        << ",fx=" << state.f << ",gx=" << state.convergence_criterion()
        << ",status=" << state.m_status << ",vd_error=" << vd_error << (better ? "(+)." : "(-).");

    if (better)
    {
        m_bias = bias;
        m_weights = weights;
        best_vd_error = vd_error;
    }

    return vd_error;
}

void linear_model_t::save(const string_t& filepath) const
{
    log_info() << "saving linear model to '" << filepath << "'...";

    std::ofstream os(filepath, std::ios::binary);
    critical(!os.is_open(),
        "failed to open the file");

    critical(
        !::nano::detail::write(os, static_cast<int32_t>(nano::major_version)) ||
        !::nano::detail::write(os, static_cast<int32_t>(nano::minor_version)) ||
        !::nano::detail::write(os, static_cast<int32_t>(nano::patch_version)) ||
        !::nano::write(os, m_bias) ||
        !::nano::write(os, m_weights),
        "failed to write to file");
}

void linear_model_t::load(const string_t& filepath)
{
    log_info() << "loading linear model from '" << filepath << "'...";

    std::ifstream is(filepath, std::ios::binary);
    critical(!is.is_open(),
        "failed to open the file");

    int32_t major{}, minor{}, patch{};
    critical(
        !::nano::detail::read(is, major) ||
        !::nano::detail::read(is, minor) ||
        !::nano::detail::read(is, patch),
        "failed to read from file");

    critical(
        major > nano::major_version ||
        (major == nano::major_version && minor > nano::minor_version) ||
        (major == nano::major_version && minor == nano::minor_version && patch > nano::patch_version),
        "version mismatch");

    critical(
        !::nano::read(is, m_bias) ||
        !::nano::read(is, m_weights),
        "failed to read from file");

    critical(m_bias.size() != m_weights.cols(),
        "parameters mismatch");
}

void linear_model_t::predict(const tensor4d_cmap_t& inputs, tensor4d_t& outputs) const
{
    ::nano::linear::predict(inputs, m_weights, m_bias, outputs);
}

void linear_model_t::predict(const tensor4d_cmap_t& inputs, tensor4d_map_t&& outputs) const
{
    ::nano::linear::predict(inputs, m_weights, m_bias, std::move(outputs));
}

void linear_model_t::evaluate(const iterator_t& iterator, const fold_t& fold, const tensor_size_t batch,
    const loss_t& loss, const tensor2d_cmap_t& weights, const tensor1d_cmap_t& bias,
    tensor1d_map_t&& errors)
{
    assert(errors.size() == iterator.samples(fold));

    ::nano::linear::iterate(iterator, fold, batch, weights, bias, [&] (
        const tensor4d_t&, const tensor4d_t& targets, const tensor4d_t& outputs,
        const tensor_size_t begin, const tensor_size_t end, const size_t)
    {
        loss.error(targets, outputs, errors.range(begin, end - begin));
    });
}

void linear_model_t::evaluate(const iterator_t& iterator, const fold_t& fold, const tensor_size_t batch,
    const loss_t& loss, const tensor2d_cmap_t& weights, const tensor1d_cmap_t& bias,
    tensor1d_map_t&& values, tensor1d_map_t&& errors)
{
    assert(values.size() == iterator.samples(fold));
    assert(errors.size() == iterator.samples(fold));

    ::nano::linear::iterate(iterator, fold, batch, weights, bias, [&] (
        const tensor4d_t&, const tensor4d_t& targets, const tensor4d_t& outputs,
        const tensor_size_t begin, const tensor_size_t end, const size_t)
    {
        loss.value(targets, outputs, values.range(begin, end - begin));
        loss.error(targets, outputs, errors.range(begin, end - begin));
    });
}
