#include <fstream>
#include <nano/tune.h>
#include <nano/logger.h>
#include <nano/version.h>
#include <nano/linear/util.h>
#include <nano/linear/model.h>
#include <nano/tensor/stream.h>
#include <nano/linear/function.h>

using namespace nano;

namespace
{
    void evaluate(const dataset_t& dataset, fold_t fold, tensor_size_t batch,
        const loss_t& loss, const tensor2d_cmap_t& weights, const tensor1d_cmap_t& bias,
        tensor1d_map_t&& errors)
    {
        assert(errors.size() == dataset.samples(fold));

        dataset.loop(execution::par, fold, batch, [&] (tensor_range_t range, size_t)
        {
            const auto inputs = dataset.inputs(fold, range);
            const auto targets = dataset.targets(fold, range);

            tensor4d_t outputs;
            ::nano::linear::predict(inputs, weights, bias, outputs);
            loss.error(targets, outputs, errors.slice(range));
        });
    }

    std::tuple<scalar_t, tensor1d_t, tensor2d_t> train(
        const linear_function_t& function, const solver_t& solver,
        tensor1d_t& tr_errors, tensor1d_t& vd_errors, train_curve_t& curve)
    {
        auto state = solver.minimize(function, vector_t::Zero(function.size()));
        auto bias = function.bias(state.x);
        auto weights = function.weights(state.x);

        // NB: rescale the bias and the weights to match the normalization of the inputs!
        const auto& istats = function.istats();
        istats.upscale(function.normalization(), weights, bias);

        const auto& loss = function.loss();
        const auto& dataset = function.dataset();

        auto fold = function.fold();
        evaluate(dataset, fold, function.batch(), loss, weights, bias, tr_errors.tensor());

        fold.m_protocol = protocol::valid;
        evaluate(dataset, fold, function.batch(), loss, weights, bias, vd_errors.tensor());

        const auto tr_value = state.f;
        const auto tr_error = tr_errors.vector().mean();
        const auto vd_error = vd_errors.vector().mean();

        curve.add(tr_value, tr_error, vd_error);
        const auto status = curve.check(1);

        log_info() << std::setprecision(8) << std::fixed
            << "fold=" << (fold.m_index + 1) << "|" << dataset.folds()
            << ":tr=" << tr_value << "|" << tr_error << ",vd=" << vd_error << "(" << status << ")"
            << ",l1reg=" << function.l1reg()
            << ",l2reg=" << function.l2reg()
            << ",vAreg=" << function.vAreg() << "," << state << ".";

        return std::make_tuple(vd_error, std::move(bias), std::move(weights));
    }
}

train_result_t linear_model_t::train(const loss_t& loss, const dataset_t& dataset, const solver_t& solver)
{
    log_info() << "training linear model...";

    tensor1d_t avg_bias(nano::size(dataset.tdim()));
    tensor2d_t avg_weights(nano::size(dataset.idim()), nano::size(dataset.tdim()));

    avg_bias.zero();
    avg_weights.zero();

    // train a model for each fold ...
    train_result_t results;
    for (size_t fold = 0, folds = dataset.folds(); fold < folds; ++ fold)
    {
        auto& result = results.add();

        const auto tr_fold = fold_t{fold, protocol::train};
        const auto vd_fold = fold_t{fold, protocol::valid};
        const auto te_fold = fold_t{fold, protocol::test};

        tensor1d_t tr_errors(dataset.samples(tr_fold));
        tensor1d_t vd_errors(dataset.samples(vd_fold));
        tensor1d_t te_errors(dataset.samples(te_fold));

        auto function = linear_function_t{loss, dataset, tr_fold};
        function.batch(batch());
        function.normalization(normalization());

        // tune the regularization factors (if any)
        tensor1d_t bias;
        tensor2d_t weights;
        scalar_t l1reg = 0.0, l2reg = 0.0, vAreg = 0.0;
        scalar_t vd_error = std::numeric_limits<scalar_t>::max();

        switch (regularization())
        {
        case ::nano::regularization::lasso:
            std::tie(l1reg, vd_error, bias, weights) = nano::grid_tune(
                nano::pow10_space_t{-6.0, +6.0},
                [&] (const scalar_t l1reg)
                {
                    function.l1reg(l1reg);
                    auto& curve = result.add(scat("l1reg=", l1reg));
                    return ::train(function, solver, tr_errors, vd_errors, curve);
                },
                tune_trials(), tune_steps());
            break;

        case ::nano::regularization::ridge:
            std::tie(l2reg, vd_error, bias, weights) = nano::grid_tune(
                nano::pow10_space_t{-6.0, +6.0},
                [&] (const scalar_t l2reg)
                {
                    function.l2reg(l2reg);
                    auto& curve = result.add(scat("l2reg=", l2reg));
                    return ::train(function, solver, tr_errors, vd_errors, curve);
                },
                tune_trials(), tune_steps());
            break;

        case ::nano::regularization::variance:
            std::tie(vAreg, vd_error, bias, weights) = nano::grid_tune(
                nano::pow10_space_t{-6.0, +6.0},
                [&] (const scalar_t vAreg)
                {
                    function.vAreg(vAreg);
                    auto& curve = result.add(scat("vAreg=", vAreg));
                    return ::train(function, solver, tr_errors, vd_errors, curve);
                },
                tune_trials(), tune_steps());
            break;

        case ::nano::regularization::elastic:
            std::tie(l1reg, l2reg, vd_error, bias, weights) = nano::grid_tune(
                nano::pow10_space_t{-8.0, +6.0},
                nano::pow10_space_t{-8.0, +6.0},
                [&] (const scalar_t l1reg, const scalar_t l2reg)
                {
                    function.l1reg(l1reg);
                    function.l2reg(l2reg);
                    auto& curve = result.add(scat("l1reg=", l1reg, ",l2reg=", l2reg));
                    return ::train(function, solver, tr_errors, vd_errors, curve);
                },
                tune_trials(), tune_steps());
            break;

        case ::nano::regularization::none:
            {
                auto& curve = result.add(scat("noreg"));
                std::tie(vd_error, bias, weights) = ::train(function, solver, tr_errors, vd_errors, curve);
            }
            break;

        default:
            critical(true, "unhandled regularization method when training linear models");
        }

        // update the average model
        avg_bias.vector() = (avg_bias.vector() * fold + bias.vector()) / (fold + 1);
        avg_weights.vector() = (avg_weights.vector() * fold + weights.vector()) / (fold + 1);

        // evaluate the current tuned model
        evaluate(dataset, te_fold, batch(), loss, weights, bias, te_errors.tensor());
        result.test(te_errors.vector().mean());
        assert(std::fabs(vd_error - result.vd_error()) < 1e-8);

        // evaluate the averaged model
        evaluate(dataset, te_fold, batch(), loss, avg_weights, avg_bias, te_errors.tensor());
        result.avg_test(te_errors.vector().mean());

        log_info() << std::setprecision(8) << std::fixed << ">>> "
            << ":tr=" << result.tr_value() << "|" << result.tr_error()
            << ",vd=" << result.vd_error()
            << ",te=" << result.te_error() << ",avg_te=" << result.avg_te_error()
            << ",l1=" << l1reg << ",l2=" << l2reg << ",vA=" << vAreg << ".";
    }

    // NB: the final model is the average across all folds!
    m_bias = avg_bias;
    m_weights = avg_weights;
    return results;
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

void linear_model_t::predict(const dataset_t& dataset, fold_t fold, tensor4d_t& outputs) const
{
    outputs.resize(cat_dims(dataset.samples(fold), dataset.tdim()));
    predict(dataset, fold, outputs.tensor());
}

void linear_model_t::predict(const dataset_t& dataset, fold_t fold, tensor4d_map_t&& outputs) const
{
    assert(outputs.dims() == cat_dims(dataset.samples(fold), dataset.tdim()));

    dataset.loop(execution::par, fold, batch(), [&] (tensor_range_t range, const size_t)
    {
        const auto inputs = dataset.inputs(fold, range);

        ::nano::linear::predict(inputs, m_weights, m_bias, outputs.slice(range));
    });
}
