#pragma once

#include <limits>
#include <nano/arch.h>
#include <unordered_map>
#include <nano/scalar.h>
#include <nano/string.h>

namespace nano
{
    using scalars_t = std::vector<scalar_t>;

    ///
    /// \brief measurement at a training point (e.g. epoch, iteration, boosting round)
    ///     for both the training and the validation datasets.
    ///
    class NANO_PUBLIC train_point_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        train_point_t() = default;

        ///
        /// \brief constructor
        ///
        train_point_t(scalar_t tr_value, scalar_t tr_error, scalar_t vd_error);

        ///
        /// \brief check if divergence occured.
        ///
        [[nodiscard]] bool valid() const;

        ///
        /// \brief access functions.
        ///
        [[nodiscard]] auto tr_value() const { return m_tr_value; }
        [[nodiscard]] auto tr_error() const { return m_tr_error; }
        [[nodiscard]] auto vd_error() const { return m_vd_error; }

    private:

        static constexpr auto i = std::numeric_limits<scalar_t>::infinity();

        // attributes
        scalar_t    m_tr_value{i}, m_tr_error{i};   ///< loss value & average error (training)
        scalar_t    m_vd_error{i};                  ///< average error (validation)
    };

    inline bool operator<(const train_point_t& lhs, const train_point_t& rhs)
    {
        return  (lhs.valid() ? lhs.vd_error() : std::numeric_limits<scalar_t>::max()) <
                (rhs.valid() ? rhs.vd_error() : std::numeric_limits<scalar_t>::max());
    }

    ///
    /// \brief
    ///
    enum class train_status
    {
        worse,          ///< the validation error has increased (compared to the previous training point)
        better,         ///< the validation error has decreased
        overfit,        ///< the validation error hasn't decreased in the given number of past training points
        diverged,       ///< training has diverged
    };

    template <>
    inline enum_map_t<train_status> enum_string<train_status>()
    {
        return
        {
            { train_status::worse,      "worse" },
            { train_status::better,     "better" },
            { train_status::overfit,    "overfit" },
            { train_status::diverged,   "diverged" },
        };
    }

    ///
    /// \brief measurements at different training points (e.g. epoch, iteration, boosting round)
    ///     for both the training and the validation datasets.
    ///
    class NANO_PUBLIC train_curve_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        train_curve_t() = default;

        ///
        /// \brief add a new measurement for the training and the validation datasets.
        ///
        void add(scalar_t tr_value, scalar_t tr_error, scalar_t vd_error);

        ///
        /// \brief check if training is done:
        ///     - either divergence is detected
        ///     - or the validation error hasn't improved in the past 'patience' steps.
        ///
        [[nodiscard]] train_status check(size_t patience) const;

        ///
        /// \brief returns the index of the optimum training point.
        ///
        [[nodiscard]] size_t optindex() const;

        ///
        /// \brief returns the optimum training point.
        ///
        [[nodiscard]] train_point_t optimum() const;

        ///
        /// \brief export to CSV with the following structure:
        ///     step,tr_value,tr_error,vd_error
        ///     0,tr_value0,tr_error0,vd_error0
        ///     1,tr_value1,tr_error1,vd_error1
        ///     2,tr_value2,tr_error2,vd_error2
        ///     .........................................
        ///
        /// NB: the header is optional and the delimeter character is configurable.
        ///
        std::ostream& save(std::ostream& stream, char delim=',', bool header=true) const;

        ///
        /// \brief access functions.
        ///
        [[nodiscard]] auto end() const { return m_points.end(); }
        [[nodiscard]] auto size() const { return m_points.size(); }
        [[nodiscard]] auto begin() const { return m_points.begin(); }
        [[nodiscard]] const auto& operator[](size_t index) const { return m_points.at(index); }

    private:

        using tpoints_t = std::vector<train_point_t>;

        // attributes
        tpoints_t   m_points;                       ///<
    };

    inline bool operator<(const train_curve_t& lhs, const train_curve_t& rhs)
    {
        return lhs.optimum() < rhs.optimum();
    }

    ///
    /// \brief collects training measurements for sets of hyper-parameters (to tune) and a fixed fold.
    ///
    class NANO_PUBLIC train_fold_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        train_fold_t() = default;

        ///
        /// \brief register a new set of hyper-parameters (e.g. regularization factors)
        ///     and return its associated training curve to edit.
        ///
        train_curve_t& add(const string_t& hyper);

        ///
        /// \brief returns the optimum hyper-parameters and its associated training curve.
        ///
        [[nodiscard]] std::pair<string_t, const train_curve_t> optimum() const;

        ///
        /// \brief set the measurement for testing dataset
        ///     (at the optimum point on the validation dataset).
        ///
        void test(scalar_t te_error)
        {
            m_te_error = te_error;
        }

        ///
        /// \brief set the measurement of the averaged model for testing dataset
        ///     (at the optimum point on the validation dataset).
        ///
        void avg_test(scalar_t ate_error)
        {
            m_ate_error = ate_error;
        }

        ///
        /// \brief access functions.
        ///
        [[nodiscard]] scalar_t tr_value() const;
        [[nodiscard]] scalar_t tr_error() const;
        [[nodiscard]] scalar_t vd_error() const;
        [[nodiscard]] scalar_t te_error() const { return m_te_error; }
        [[nodiscard]] scalar_t avg_te_error() const { return m_ate_error; }

    private:

        using tcurves_t = std::unordered_map<string_t, train_curve_t>;

        static constexpr auto i = std::numeric_limits<scalar_t>::infinity();

        // attributes
        tcurves_t   m_curves;                       ///<
        scalar_t    m_te_error{i};                  ///< average error (testing)
        scalar_t    m_ate_error{i};                 ///< average error of the averaged model (testing)
    };

    ///
    /// \brief collects training measurements across folds.
    ///
    class NANO_PUBLIC train_result_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        train_result_t() = default;

        ///
        /// \brief register a new fold and return its associated training session to edit.
        ///
        auto& add()
        {
            m_folds.emplace_back();
            return *m_folds.rbegin();
        }

        ///
        /// \brief export to CSV with the following structure:
        ///     fold,tr_error,vd_error,te_error,avg_te_error
        ///     0,tr_error0,vd_error0,te_error0,avg_te_error0
        ///     1,tr_error1,vd_error2,te_error1,avg_te_error1
        ///     2,tr_error2,vd_error2,te_error2,avg_te_error2
        ///     .........................................
        ///
        /// NB: the header is optional and the delimeter character is configurable.
        ///
        std::ostream& save(std::ostream& stream, char delim=',', bool header=true) const;

        ///
        /// \brief access functions.
        ///
        [[nodiscard]] auto end() const { return m_folds.end(); }
        [[nodiscard]] auto size() const { return m_folds.size(); }
        [[nodiscard]] auto begin() const { return m_folds.begin(); }
        [[nodiscard]] const auto& operator[](size_t index) const { return m_folds.at(index); }

    private:

        using tfolds_t = std::vector<train_fold_t>;

        // attributes
        tfolds_t    m_folds;                        ///<
    };
}
