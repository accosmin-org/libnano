#pragma once

#include <nano/arch.h>
#include <nano/json.h>
#include <nano/tensor.h>
#include <nano/factory.h>
#include <nano/string_utils.h>

namespace nano
{
    class dataset_t;
    using dataset_factory_t = factory_t<dataset_t>;
    using rdataset_t = dataset_factory_t::trobject;

    ///
    /// \brief dataset splitting protocol.
    ///
    enum class protocol
    {
        train = 0,      ///< training
        valid,          ///< validation (for tuning hyper-parameters)
        test            ///< testing
    };

    template <>
    inline enum_map_t<protocol> enum_string<protocol>()
    {
        return
        {
            { protocol::train,    "train" },
            { protocol::valid,    "valid" },
            { protocol::test,     "test" }
        };
    }

    ///
    /// \brief dataset splitting fold.
    ///
    struct fold_t
    {
        size_t          m_index;        ///< fold index
        protocol        m_protocol;     ///<
    };

    inline bool operator==(const fold_t& f1, const fold_t& f2)
    {
        return f1.m_index == f2.m_index && f1.m_protocol == f2.m_protocol;
    }

    inline bool operator<(const fold_t& f1, const fold_t& f2)
    {
        return f1.m_index < f2.m_index || (f1.m_index == f2.m_index && f1.m_protocol < f2.m_protocol);
    }

    ///
    /// \brief input feature.
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
        feature_t(string_t name, strings_t categories) :
            m_name(std::move(name)), m_categories(std::move(categories)) {}

        ///
        /// \brief returns if the feature is continuous
        ///
        bool continuous() const { return m_categories.empty(); }

        ///
        /// \brief returns if the feature is categorical
        ///
        bool categorical() const { return !continous(); }

        ///
        /// \brief access functions
        ///
        const auto& name() const { return m_name; }
        const auto& categories() const { return m_categories; }

    private:

        // attributes
        string_t    m_name;         ///<
        strings_t   m_categories;   ///< all possible categories if the feature is categorical
    };

    ///
    /// \brief machine learning dataset consisting of a collection of fixed-size 3D input tensors
    ///     split into training, validation and testing parts.
    ///
    /// dataset_t's interface handles the following cases:
    ///     - both supervised and unsupervised machine learning tasks
    ///     - categorical and continuous features
    ///     - missing feature values
    ///
    class NANO_PUBLIC dataset_t : public json_configurable_t
    {
    public:

        ///
        /// \brief returns the available implementations
        ///
        static dataset_factory_t& all();

        ///
        /// \brief populate the dataset with samples
        ///
        virtual bool load() = 0;

        ///
        /// \brief returns the number of folds
        ///
        virtual tensor_size_t folds() const = 0;

        ///
        /// \brief returns the number of samples of a given fold
        ///
        virtual tensor_size_t samples(const fold_t& fold) const = 0;

        ///
        /// \brief returns the number of samples having a given label in a given fold
        ///
        virtual tensor_size_t samples(const fold_t& fold, const string_t& label) const = 0;

        ///
        /// \brief returns the total number of labels (if a classification task)
        ///
        virtual tensor_size_t labels() const = 0;

        ///
        /// \brief returns the label for a sample in the given fold
        ///
        virtual string_t label(const fold_t& fold, const tensor_size_t index) const = 0;

        ///
        /// \brief returns the total number of features
        ///
        virtual tensor_size_t features() const = 0;

        ///
        /// \brief returns information about the given feature
        ///
        virtual feature_t feature(const tensor_size_t index) const = 0;

        ///
        /// \brief returns the size of the input (aka feature values) tensor
        ///
        virtual tensor3d_dim_t input_dims() const = 0;

        ///
        /// \brief returns the input tensor for a sample in the given fold
        ///
        virtual tensor3d_t input(const fold_t& fold, const tensor_size_t index) const = 0;

        ///
        /// \brief returns the hash of the input tensor for a sample in the given fold
        ///
        virtual size_t input_hash(const fold_t&, const tensor_size_t index) const = 0;

        ///
        /// \brief returns the size of the target tensor (if a supervised task)
        ///
        virtual tensor3d_dim_t target_dims() const = 0;

        ///
        /// \brief returns the target tensor for a sample in the given fold (if a supervised task)
        ///
        virtual tensor3d_t target(const fold_t& fold, const tensor_size_t index) const = 0;

        ///
        /// \brief returns true if an input value is present
        ///
        static bool missing(const scalar_t input_value) { return !std::isfinite(input_value); }

        ///
        /// \brief returns true if the dataset is appropriate for supervised machine learning tasks
        ///     (aka if the targets are present)
        ///
        bool supervised() const { return nano::size(target_dims()) > 0; }

        ///
        /// \brief randomly shuffle the samples associated for the given fold
        ///
        virtual void shuffle(const fold_t&) const = 0;

        ///
        /// \brief print a short description of the dataset
        ///
        void describe(const string_t& name) const;

        ///
        /// \brief returns the number of duplicated samples of the given fold
        ///
        size_t duplicates(const fold_t&) const;

        ///
        /// \brief returns the number of common samples between
        ///     training, validation and test splits of the given fold index
        ///
        size_t intersections(const size_t fold_index) const;
    };
}
