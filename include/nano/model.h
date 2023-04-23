#pragma once

#include <nano/learner.h>
#include <nano/loss.h>
#include <nano/model/result.h>
#include <nano/solver.h>
#include <nano/splitter.h>
#include <nano/tuner.h>

namespace nano
{
class model_t;
using rmodel_t = std::unique_ptr<model_t>;

///
/// \brief interface for machine learning models.
///
/// the minimum set of operations are:
///     - training (mutable) which fits the model on the given dataset,
///     - prediction (constant) which evaluates the trained model on the given dataset,
///     - saving/reading to/from binary streams.
///
class NANO_PUBLIC model_t : public learner_t, public clonable_t<model_t>
{
public:
    ///
    /// \brief logging operator: op(fit_result, prefix)
    ///
    using logger_t = std::function<void(const fit_result_t&, const string_t&)>;

    ///
    /// \brief returns a default logging implementation that prints the current status to standard I/O.
    ///
    static logger_t make_logger_stdio(int precision = 8);

    ///
    /// \brief default constructor.
    ///
    explicit model_t(string_t id);

    ///
    /// \brief returns the available implementations.
    ///
    static factory_t<model_t>& all();

    ///
    /// \brief fit the model using the given samples and the current set of (hyper-)parameters
    ///     and returns the associated statistics.
    ///
    virtual fit_result_t fit(const dataset_t&, const indices_t&, const loss_t&, const solver_t&, const splitter_t&,
                             const tuner_t&) = 0;

    ///
    /// \brief evaluate the trained model and returns the predictions for each of the given samples.
    ///
    virtual tensor4d_t predict(const dataset_t&, const indices_t&) const = 0;

    ///
    /// \brief set the logging callback.
    ///
    virtual void logger(logger_t logger);

protected:
    void log(const fit_result_t& fit_result) const;

    auto make_logger_lambda() const
    {
        return [this](const auto& result) { log(result); };
    }

private:
    // attributes
    logger_t m_logger;
};
} // namespace nano
