#pragma once

#include <nano/json.h>
#include <nano/tensor.h>
#include <nano/string.h>

namespace nano
{
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
        feature_t(string_t&& name) :
            m_name(std::move(name))
        {
        }

        ///
        /// \brief creates and returns a continuous feature using the given parameters
        ///
        static auto make_scalar(string_t name, string_t placeholder = string_t())
        {
            auto feature = feature_t{std::move(name)};
            feature.m_placeholder = std::move(placeholder);
            return feature;
        }

        ///
        /// \brief creates and returns a categorical feature using the given parameters
        ///
        static auto make_discrete(string_t name, strings_t labels, string_t placeholder = string_t())
        {
            auto feature = feature_t{std::move(name)};
            feature.m_labels = std::move(labels);
            feature.m_placeholder = std::move(placeholder);
            return feature;
        }

        ///
        /// \brief serialize to JSON
        ///
        json_t config() const
        {
            json_t json;
            json["name"] = m_name;
            json["placeholder"] = m_placeholder;
            auto&& labels = (json["labels"] = json_t::array());
            for (const auto& label : m_labels)
            {
                labels.push_back(label);
            }
            return json;
        }

        ///
        /// \brief deserialize from JSON
        ///
        void config(const json_t& json)
        {
            m_name = json["name"];
            m_placeholder = json["placeholder"];
            m_labels.clear();
            for (const auto& label : json["labels"])
            {
                m_labels.push_back(label.get<string_t>());
            }
        }

        ///
        /// \brief returns true if the feature is discrete
        ///
        bool discrete() const { return !m_labels.empty(); }

        ///
        /// \brief returns true if the feature is optional
        ///
        bool optional() const { return !m_placeholder.empty(); }

        ///
        /// \brief returns the value to store when the feature value is missing
        ///
        static auto placeholder_value()
        {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        }

        ///
        /// \brief returns true if the given stored value indicates that the feature value is missing
        ///
        static bool missing(const scalar_t value) { return !std::isfinite(value); }

        ///
        /// \brief returns the label associated to the given feature value (if possible)
        ///
        auto label(const scalar_t value) const
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
        /// \brief access functions
        ///
        const auto& name() const { return m_name; }
        const auto& labels() const { return m_labels; }
        const auto& placeholder() const { return m_placeholder; }

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
    /// \brief print a description of the given feature.
    ///
    inline std::ostream& operator<<(std::ostream& os, const feature_t& f)
    {
        os << "name=" << f.name() << ",labels[";
        for (const auto& label : f.labels())
        {
            os << label;
            if (&label != &(*(f.labels().rbegin())))
            {
                os << ",";
            }
        }
        return os << "],placeholder=" << f.placeholder();
    }
}
