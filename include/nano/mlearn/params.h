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
class NANO_PUBLIC params_t
{
public:
    ///
    /// \brief default constructor
    ///
    params_t();

    ///
    /// \brief enable copying
    ///
    params_t(const params_t&);
    params_t& operator=(const params_t&);

    ///
    /// \brief enable copying
    ///
    params_t(params_t&&) noexcept;
    params_t& operator=(params_t&&) noexcept;

    ///
    /// \brief default destructor
    ///
    ~params_t();

    ///
    /// \brief change the tuning strategy.
    ///
    params_t& tuner(rtuner_t&&);
    params_t& tuner(const tuner_t&);
    params_t& tuner(const rtuner_t&);
    params_t& tuner(const string_t& id);

    ///
    /// \brief change the numerical optimization method.
    ///
    params_t& solver(rsolver_t&&);
    params_t& solver(const solver_t&);
    params_t& solver(const rsolver_t&);
    params_t& solver(const string_t& id);

    ///
    /// \brief change the sample splitting strategy.
    ///
    params_t& splitter(rsplitter_t&&);
    params_t& splitter(const splitter_t&);
    params_t& splitter(const rsplitter_t&);
    params_t& splitter(const string_t& id);

    ///
    /// \brief change the logging method.
    ///
    params_t& logger(logger_t);

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
    void log(const result_t&, tensor_size_t last_trial, const string_t& prefix, int precision = 8) const;

private:
    // attributes
    logger_t    m_logger;   ///<
    rtuner_t    m_tuner;    ///<
    rsolver_t   m_solver;   ///<
    rsplitter_t m_splitter; ///<
};
} // namespace nano::ml
