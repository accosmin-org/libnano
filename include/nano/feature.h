#pragma once

#include <cmath>
#include <nano/arch.h>
#include <nano/core/strutil.h>
#include <nano/task.h>
#include <nano/tensor.h>

namespace nano
{
class feature_t;
using features_t = std::vector<feature_t>;

///
/// \brief input or target feature type.
///
enum class feature_type : uint8_t
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
    sclass, ///< categorical feature (single-label - one value possible out of a fixed set)
    mclass, ///< categorical feature (mulit-label - a subset of values possible out of a fixed set)
};

template <>
inline enum_map_t<feature_type> enum_string()
{
    return {
        {   feature_type::int8,    "int8"},
        {  feature_type::int16,   "int16"},
        {  feature_type::int32,   "int32"},
        {  feature_type::int64,   "int64"},
        {  feature_type::uint8,   "uint8"},
        { feature_type::uint16,  "uint16"},
        { feature_type::uint32,  "uint32"},
        { feature_type::uint64,  "uint64"},
        {feature_type::float32, "float32"},
        {feature_type::float64, "float64"},
        { feature_type::sclass,  "sclass"},
        { feature_type::mclass,  "mclass"}
    };
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
    size_t set_label(const char* label) const;
    size_t set_label(const string_t& label) const;
    size_t set_label(const std::string_view& label) const;

    ///
    /// \brief returns true if the feature is valid (aka defined).
    ///
    bool valid() const;

    ///
    /// \brief returns the associated machine learning task if this feature is the target.
    ///
    task_type task() const;

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
    /// \brief returns the feature type.
    ///
    feature_type type() const { return m_type; }

    ///
    /// \brief returns the tensor dimensions (useful if a continuous feature).
    ///
    tensor3d_dims_t dims() const { return m_dims; }

    ///
    /// \brief returns the feature name.
    ///
    const string_t& name() const { return m_name; }

    ///
    /// \brief returns the set of labels (useful if a categorical feature).
    ///
    const strings_t& labels() const { return m_labels; }

    ///
    /// \brief returns the number of classes (useful if a categorical feature).
    ///
    tensor_size_t classes() const { return static_cast<tensor_size_t>(m_labels.size()); }

    ///
    /// \brief returns true if a single-label categorical feature.
    ///
    bool is_sclass() const { return valid() && m_type == feature_type::sclass; }

    ///
    /// \brief returns true if a multi-label categorical feature.
    ///
    bool is_mclass() const { return valid() && m_type == feature_type::mclass; }

    ///
    /// \brief returns true if a scalar continuous feature.
    ///
    bool is_scalar() const
    {
        return valid() && (m_type != feature_type::sclass && m_type != feature_type::mclass) && size(m_dims) == 1;
    }

    ///
    /// \brief returns true if a structured continuous feature.
    ///
    bool is_struct() const
    {
        return valid() && (m_type != feature_type::sclass && m_type != feature_type::mclass) && size(m_dims) > 1;
    }

private:
    // attributes
    feature_type      m_type{feature_type::float32}; ///<
    tensor3d_dims_t   m_dims{1, 1, 1};               ///< dimensions (if continuous)
    string_t          m_name;                        ///<
    mutable strings_t m_labels;                      ///< possible labels (if the feature is discrete/categorical)
};

NANO_PUBLIC bool operator==(const feature_t& lhs, const feature_t& rhs);
NANO_PUBLIC bool operator!=(const feature_t& lhs, const feature_t& rhs);
NANO_PUBLIC std::ostream& operator<<(std::ostream&, const feature_t&);
} // namespace nano
