#pragma once

#include <cmath>
#include <nano/arch.h>
#include <nano/tensor.h>
#include <nano/dataset/task.h>

namespace nano
{
    class feature_t;
    using features_t = std::vector<feature_t>;

    ///
    /// \brief input or target feature type.
    ///
    enum class feature_type : int32_t
    {
        // continuous features
        int8 = 0,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float32,
        float64,

        // discrete features
        sclass,         ///< categorical feature (single-label - one value possible out of a fixed set)
        mclass,         ///< categorical feature (mulit-label - a subset of values possible out of a fixed set)
    };

    template <>
    inline enum_map_t<feature_type> enum_string<feature_type>()
    {
        return
        {
            { feature_type::int8,       "int8" },
            { feature_type::int16,      "int16" },
            { feature_type::int32,      "int32" },
            { feature_type::int64,      "int64" },
            { feature_type::uint8,      "uint8" },
            { feature_type::uint16,     "uint16" },
            { feature_type::uint32,     "uint32" },
            { feature_type::uint64,     "uint64" },
            { feature_type::float32,    "float32" },
            { feature_type::float64,    "float64" },
            { feature_type::sclass,     "sclass" },
            { feature_type::mclass,     "mclass" },
        };
    }

    inline std::ostream& operator<<(std::ostream& stream, feature_type value)
    {
        return stream << scat(value);
    }

    ///
    /// \brief input feature (e.g. describes a column in a csv file)
    ///     that can be either discrete/categorical or scalar/continuous
    ///     and with or without missing values.
    ///
    class NANO_PUBLIC feature_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        feature_t();

        ///
        /// \brief constructor
        ///
        explicit feature_t(string_t name);

        ///
        /// \brief set the feature as continuous.
        ///
        feature_t& scalar(feature_type type = feature_type::float32, tensor3d_dims_t dims = make_dims(1, 1, 1));

        ///
        /// \brief set the feature as discrete, by passing the labels.
        /// NB: this is useful when the labels are known before loading some dataset.
        ///
        feature_t& sclass(strings_t labels);
        feature_t& mclass(strings_t labels);

        ///
        /// \brief set the feature as discrete, but the labels are not known.
        /// NB: this is useful when the labels are discovered while loading some dataset.
        ///
        feature_t& sclass(size_t count);
        feature_t& mclass(size_t count);

        ///
        /// \brief try to add the given label if possible.
        /// NB: this is useful when the labels are discovered while loading some dataset.
        ///
        size_t set_label(const string_t& label) const;

        ///
        /// \brief returns true if the feature is valid (aka defined).
        ///
        operator bool() const // NOLINT(hicpp-explicit-conversions)
        {
            return !m_name.empty();
        }

        ///
        /// \brief returns the associated machine learning task if this feature is the target.
        ///
        explicit operator task_type() const;

        ///
        /// \brief serialize from the given binary stream.
        ///
        /// NB: any error is considered critical and expected to result in an exception.
        ///
        std::istream& read(std::istream&);

        ///
        /// \brief serialize to the given binary stream.
        ///
        /// NB: any error is considered critical and expected to result in an exception.
        ///
        std::ostream& write(std::ostream&) const;

        ///
        /// \brief access functions
        ///
        auto type() const { return m_type; }
        const auto& dims() const { return m_dims; }
        const auto& name() const { return m_name; }
        const auto& labels() const { return m_labels; }
        auto classes() const { return static_cast<tensor_size_t>(m_labels.size()); }

    private:

        // attributes
        feature_type        m_type{feature_type::float32};  ///<
        tensor3d_dims_t     m_dims{1, 1, 1};        ///< dimensions (if continuous)
        string_t            m_name;                 ///<
        mutable strings_t   m_labels;               ///< possible labels (if the feature is discrete/categorical)
    };

    NANO_PUBLIC bool operator==(const feature_t& f1, const feature_t& f2);
    NANO_PUBLIC bool operator!=(const feature_t& f1, const feature_t& f2);
    NANO_PUBLIC std::ostream& operator<<(std::ostream& stream, const feature_t& feature);

    NANO_PUBLIC std::istream& read(std::istream& stream, feature_t&);
    NANO_PUBLIC std::ostream& write(std::ostream& stream, const feature_t&);

    ///
    /// \brief describe a feature (e.g. as selected by a weak learner) in terms of
    ///     e.g. importance (impact on error rate).
    ///
    class feature_info_t;
    using feature_infos_t = std::vector<feature_info_t>;

    class NANO_PUBLIC feature_info_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        feature_info_t();

        ///
        /// \brief constructor
        ///
        feature_info_t(tensor_size_t feature, tensor_size_t count, scalar_t importance);

        ///
        /// \brief sort a list of (selected) features by their index.
        ///
        static void sort_by_index(feature_infos_t& features);

        ///
        /// \brief sort a list of (selected) features descendingly by their importance.
        ///
        static void sort_by_importance(feature_infos_t& features);

        ///
        /// \brief change the feature's importance.
        ///
        void importance(scalar_t importance);

        ///
        /// \brief access functions
        ///
        auto count() const { return m_count; }
        auto feature() const { return m_feature; }
        auto importance() const { return m_importance; }

    private:

        // attributes
        tensor_size_t   m_feature{-1};      ///< feature index
        tensor_size_t   m_count{0};         ///< how many times it was selected (e.g. folds)
        scalar_t        m_importance{0.0};  ///< feature importance (e.g. impact on performance)
    };
}
