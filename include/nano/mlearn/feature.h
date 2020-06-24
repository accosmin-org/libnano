#pragma once

#include <cmath>
#include <nano/scalar.h>
#include <nano/string.h>

namespace nano
{
    class feature_t;
    using features_t = std::vector<feature_t>;

    ///
    /// \brief input feature (e.g. describes a column in a CSV file)
    ///     that can be either discrete/categorical or scalar/continuous
    ///     and with or without missing values.
    ///
    class feature_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        feature_t() = default;

        ///
        /// \brief constructor
        ///
        explicit feature_t(string_t name) :
            m_name(std::move(name))
        {
        }

        ///
        /// \brief set the placeholder (the feature becomes optional).
        ///
        auto& placeholder(string_t placeholder)
        {
            m_placeholder = std::move(placeholder);
            return *this;
        }

        ///
        /// \brief set the labels (the feature become discrete).
        ///
        auto& labels(strings_t labels)
        {
            m_labels = std::move(labels);
            return *this;
        }

        ///
        /// \brief returns true if the feature is discrete.
        ///
        [[nodiscard]] bool discrete() const
        {
            return !m_labels.empty();
        }

        ///
        /// \brief returns true if the feature is optional.
        ///
        [[nodiscard]] bool optional() const
        {
            return !m_placeholder.empty();
        }

        ///
        /// \brief returns the value to store when the feature value is missing.
        ///
        static auto placeholder_value()
        {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        }

        ///
        /// \brief returns true if the given stored value indicates that the feature value is missing.
        ///
        static bool missing(const scalar_t value)
        {
            return !std::isfinite(value);
        }

        ///
        /// \brief returns the label associated to the given feature value (if possible).
        ///
        [[nodiscard]] auto label(const scalar_t value) const
        {
            if (!discrete())
            {
                throw std::invalid_argument("labels are only available for discrete features");
            }
            else
            {
                return missing(value) ? string_t() : m_labels.at(static_cast<size_t>(value));
            }
        }

        ///
        /// \brief returns true if the feature is valid (aka defined).
        ///
        operator bool() const { return !m_name.empty(); } // NOLINT(hicpp-explicit-conversions)

        ///
        /// \brief access functions
        ///
        [[nodiscard]] const auto& name() const { return m_name; }
        [[nodiscard]] const auto& labels() const { return m_labels; }
        [[nodiscard]] const auto& placeholder() const { return m_placeholder; }

    private:

        // attributes
        string_t    m_name;         ///<
        strings_t   m_labels;       ///< possible labels (if the feature is discrete/categorical)
        string_t    m_placeholder;  ///< placeholder string used if its value is missing
    };

    ///
    /// \brief returns true if the two given feature are equivalent.
    ///
    inline bool operator==(const feature_t& f1, const feature_t& f2)
    {
        return  f1.name() == f2.name() &&
                f1.labels() == f2.labels() &&
                f1.placeholder() == f2.placeholder();
    }

    ///
    /// \brief describe a feature (e.g. as selected by a weak learner) in terms of
    ///     e.g. importance (impact on error rate).
    ///
    class feature_info_t;
    using feature_infos_t = std::vector<feature_info_t>;

    class feature_info_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        feature_info_t() = default;

        ///
        /// \brief constructor
        ///
        feature_info_t(tensor_size_t feature, tensor_size_t folds, scalar_t importance) :
            m_feature(feature),
            m_folds(folds),
            m_importance(importance)
        {
        }

        ///
        /// \brief sort a list of (selected) features by their index.
        ///
        static void sort_by_index(feature_infos_t& features)
        {
            std::stable_sort(features.begin(), features.end(), [] (const auto& lhs, const auto& rhs)
            {
                return lhs.m_feature < rhs.m_feature;
            });
        }

        ///
        /// \brief sort a list of (selected) features descendingly by their importance.
        ///
        static void sort_by_importance(feature_infos_t& features)
        {
            std::stable_sort(features.begin(), features.end(), [] (const auto& lhs, const auto& rhs)
            {
                return lhs.m_importance > rhs.m_importance;
            });
        }

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto folds() const { return m_folds; }
        [[nodiscard]] auto feature() const { return m_feature; }
        [[nodiscard]] auto importance() const { return m_importance; }

    private:

        // attributes
        tensor_size_t   m_feature{-1};      ///< feature index
        tensor_size_t   m_folds{0};         ///< in how many folds the feature was selected
        scalar_t        m_importance{0.0};  ///< feature importance (e.g. impact on performance)
    };
}
