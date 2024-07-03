#pragma once

#include <nano/solver.h>
#include <nano/splitter.h>
#include <nano/tuner.h>

namespace nano::ml
{
class result_t;

///
/// \brief utility to gather common parameters useful for fitting machine learning models.
///
/// NB: the default parameters are suitable for most machine learning tasks.
///
class NANO_PUBLIC config_t
{
public:
    // TODO: set directory where to log everything...
    //  -> log: chosen hyper-parameter values, statistics per fold for each hyper-parameter config
    //  -> hparamX/foldX/model: serialized model
    //  -> hparamX/foldX/log: detailed log - solver steps, chosen weak learners etc.

    ///
    /// \brief logging operator: op(result - up to the current step, prefix)
    ///
    using logger_t = std::function<void(const result_t&, const string_t&)>;

    ///
    /// \brief returns a default logging implementation that prints the current status to standard I/O.
    ///
    static logger_t make_stdio_logger(int precision = 8);

    ///
    /// \brief default constructor
    ///
    config_t();

    ///
    /// \brief enable copying
    ///
    config_t(const config_t&);
    config_t& operator=(const config_t&);

    ///
    /// \brief enable copying
    ///
    config_t(config_t&&) noexcept;
    config_t& operator=(config_t&&) noexcept;

    ///
    /// \brief default destructor
    ///
    ~config_t();

    ///
    /// \brief change the tuning strategy.
    ///
    config_t& tuner(rtuner_t&&);
    config_t& tuner(const tuner_t&);
    config_t& tuner(const rtuner_t&);
    config_t& tuner(const string_t& id);

    ///
    /// \brief change the numerical optimization method.
    ///
    config_t& solver(rsolver_t&&);
    config_t& solver(const solver_t&);
    config_t& solver(const rsolver_t&);
    config_t& solver(const string_t& id);

    ///
    /// \brief change the sample splitting strategy.
    ///
    config_t& splitter(rsplitter_t&&);
    config_t& splitter(const splitter_t&);
    config_t& splitter(const rsplitter_t&);
    config_t& splitter(const string_t& id);

    ///
    /// \brief change the logging method.
    ///
    config_t& logger(logger_t);

    ///
    /// \brief return the current tuning strategy.
    ///
    const tuner_t& tuner() const;

    ///
    /// \brief return the current numerical optimization method.
    ///
    const solver_t& solver() const;

    ///
    /// \brief return the current sample splitting strategy.
    ///
    const splitter_t& splitter() const;

    ///
    /// \brief return the current logging method.
    ///
    const logger_t& logger() const;

    ///
    /// \brief log the current fitting result.
    ///
    void log(const result_t&, const string_t& prefix) const;

private:
    // attributes
    logger_t    m_logger;   ///<
    rtuner_t    m_tuner;    ///<
    rsolver_t   m_solver;   ///<
    rsplitter_t m_splitter; ///<
};
} // namespace nano::ml
