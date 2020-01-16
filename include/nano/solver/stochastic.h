#pragma once

#include <nano/solver.h>
#include <nano/solver/schedule.h>

namespace nano
{
    class stochastic_solver_t;
    using stochastic_solver_factory_t = factory_t<stochastic_solver_t>;
    using rstochastic_solver_t = stochastic_solver_factory_t::trobject;

    ///
    /// \brief stochastic gradient (descent) with:
    ///     - automatic tuning of the learning rate,
    ///     - a configurable decay factor,
    ///     - a configurable minibatch size and
    ///     - a configurable factor to geometrically increase the minibatch size (aka minibatch ratio, see (1)).
    ///
    ///     see (1) "Optimization Methods forLarge-Scale Machine Learning", by L. Bottou, F. E. Curtis, J. Nocedal
    ///
    /// NB: the initial learning rate may be decreased geometrically and
    ///     the initial decay factor may be increased arithmetically and
    ///     if the function value either diverges or increases after an epoch.
    ///
    class NANO_PUBLIC stochastic_solver_t : public solver_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        stochastic_solver_t() = default;

        ///
        /// \brief returns the available implementations
        ///
        static stochastic_solver_factory_t& all();

        ///
        /// \brief change parameters
        ///
        void decay(const scalar_t decay) { m_decay = decay; }
        void batch0(const int64_t batch0) { m_batch0 = batch0; }
        void batchr(const scalar_t batchr) { m_batchr = batchr; }
        void tuneit(const int64_t tuneit) { m_tuneit = tuneit; }

        ///
        /// \brief access functions
        ///
        auto decay() const { return m_decay.get(); }
        auto batch0() const { return m_batch0.get(); }
        auto batchr() const { return m_batchr.get(); }
        auto tuneit() const { return m_tuneit.get(); }

    protected:

        static solver_state_t init_state(const function_t&);

        ///
        /// \brief tune the learning rate such that:
        ///     - the function value does not diverge and
        ///     - the learning rate is as high as possible
        ///
        lrate_schedule_t tune(const solver_function_t&, vector_t x0, solver_state_t&) const;

    private:

        // attributes
        sparam1_t   m_decay{"solver::stoch::decay", 0.0, LE, 0.5, LE, 1.0};     ///< decay factor
        iparam1_t   m_batch0{"solver::stoch::batch0", 1, LE, 1, LE, 1024};      ///< minibatch size in number of summands
        sparam1_t   m_batchr{"solver::stoch::batchr", 1.0, LE, 1.0, LE, 1.1};   ///< minibatch ratio
        iparam1_t   m_tuneit{"solver::stoch::tuneit", 100, LE, 1000, LE, 10000};///< number of iterations to use for tuning
    };

    ///
    /// \brief stochastic gradient (descent) with:
    ///     - automatic tuning of the learning rate and its associated decay factor (starting from the given hints)
    ///     - a configurable minibatch size and
    ///     - a configurable factor to geometrically increase the minibatch size (aka minibatch ratio, see (1)).
    ///
    ///     see (1) "Optimization Methods forLarge-Scale Machine Learning", by L. Bottou, F. E. Curtis, J. Nocedal
    ///
    /// NB: the initial learning rate may be decreased and the initial decay factor may be increased
    ///     if the function value either diverges or increases after an epoch.
    ///
    class NANO_PUBLIC solver_sgd_t : public stochastic_solver_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_sgd_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const override;
    };

    ///
    /// \brief stochastic gradient (descent) with averaging.
    ///
    ///     see (1) "New method of stochastic approximation type", by B. T. Polyak
    ///     see (2) "Acceleration of stochastic approximation by averaging", by B. T. Polyak, A. B. Juditsky
    ///
    /// NB: the averaging starts from step2 (@see solver_t).
    ///
    class NANO_PUBLIC solver_asgd_t final : public stochastic_solver_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_asgd_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;
    };
}
