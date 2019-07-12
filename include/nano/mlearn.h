#pragma once

#include <nano/random.h>
#include <nano/string.h>
#include <nano/tensor.h>

namespace nano
{
    ///
    /// \brief target value of the positive class.
    ///
    inline scalar_t pos_target() { return +1; }

    ///
    /// \brief target value of the negative class.
    ///
    inline scalar_t neg_target() { return -1; }

    ///
    /// \brief check if a target value maps to a positive class.
    ///
    inline bool is_pos_target(const scalar_t target) { return target > 0; }

    ///
    /// \brief target tensor for single and multi-label classification problems with [n_labels] classes.
    ///
    namespace detail
    {
        inline void class_target(tensor3d_t&)
        {
        }

        template <typename... tindices>
        inline void class_target(tensor3d_t& target, const tensor_size_t index, const tindices... indices)
        {
            if (index >= 0 && index < target.size())
            {
                target(index) = pos_target();
            }
            class_target(target, indices...);
        }
    }

    template <typename... tindices>
    inline tensor3d_t class_target(const tensor_size_t n_labels, const tindices... indices)
    {
        tensor3d_t target(n_labels, 1, 1);
        target.constant(neg_target());
        detail::class_target(target, indices...);
        return target;
    }

    ///
    /// \brief target vector for multi-label classification problems based on the sign of the predictions.
    ///
    inline tensor3d_t class_target(const tensor3d_t& outputs)
    {
        tensor3d_t target(outputs.dims());
        for (auto i = 0; i < outputs.size(); ++ i)
        {
            target(i) = is_pos_target(outputs(i)) ? pos_target() : neg_target();
        }
        return target;
    }

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
    /// \brief dataset splitting sample indices into training, validation and test.
    ///
    struct split_t
    {
        ///
        /// \brief returns the sample indices of the given fold.
        ///
        auto& indices(const fold_t& fold)
        {
            switch (fold.m_protocol)
            {
            case protocol::train:   return m_tr_indices;
            case protocol::valid:   return m_vd_indices;
            default:                return m_te_indices;
            }
        }

        ///
        /// \brief returns the sample indices of the given fold.
        ///
        const auto& indices(const fold_t& fold) const
        {
            switch (fold.m_protocol)
            {
            case protocol::train:   return m_tr_indices;
            case protocol::valid:   return m_vd_indices;
            default:                return m_te_indices;
            }
        }

        // attributes
        indices_t           m_tr_indices;   ///< indices of the training samples
        indices_t           m_vd_indices;   ///< indices of the validation samples
        indices_t           m_te_indices;   ///< indices of the test samples
    };

    ///
    /// \brief randomly split `count` elements in two disjoint sets:
    ///     the first with (`percentage1`)% elements and
    ///     the second with the remaining (100-`percentage1`)% elements.
    ///
    /// NB: the indices in each set are sorted to potentially improve speed.
    ///
    inline auto split2(const tensor_size_t count, const tensor_size_t percentage1)
    {
        assert(percentage1 >= 0 && percentage1 <= 100);

        const auto size1 = percentage1 * count / 100;
        const auto size2 = count - size1;

        indices_t all = indices_t::LinSpaced(count, 0, count);
        std::shuffle(begin(all), end(all), make_rng());

        indices_t set1 = all.segment(0, size1);
        indices_t set2 = all.segment(size1, size2);

        std::sort(begin(set1), end(set1));
        std::sort(begin(set2), end(set2));

        return std::make_tuple(set1, set2);
    }

    ///
    /// \brief randomly split `count` elements in three disjoint sets:
    ///     the first with (`percentage1`)% elements,
    ///     the second with (`percentage2`)% elements and
    ///     the third with the remaining (100-`percentage1`-`percentage2`)% elements.
    ///
    /// NB: the indices in each set are sorted to potentially improve speed.
    ///
    inline auto split3(const tensor_size_t count, const tensor_size_t percentage1, const tensor_size_t percentage2)
    {
        assert(percentage1 >= 0 && percentage1 <= 100);
        assert(percentage2 >= 0 && percentage2 <= 100);
        assert(percentage1 + percentage2 <= 100);

        const auto size1 = percentage1 * count / 100;
        const auto size2 = percentage2 * count / 100;
        const auto size3 = count - size1 - size2;

        indices_t all = indices_t::LinSpaced(count, 0, count);
        std::shuffle(begin(all), end(all), make_rng());

        indices_t set1 = all.segment(0, size1);
        indices_t set2 = all.segment(size1, size2);
        indices_t set3 = all.segment(size2, size3);

        std::sort(begin(set1), end(set1));
        std::sort(begin(set2), end(set2));
        std::sort(begin(set3), end(set3));

        return std::make_tuple(set1, set2, set3);
    }

    ///
    /// \brief randomly sample with replacement the given percentage of `count` elements.
    ///
    /// NB: the returned indices are sorted to potentially improve speed.
    ///
    inline auto sample_with_replacement(const tensor_size_t count, const tensor_size_t percentage)
    {
        assert(0 <= percentage && percentage <= 100);

        auto rng = make_rng();
        auto udist = make_udist<tensor_size_t>(0, count - 1);

        indices_t set(percentage * count / 100);
        std::generate(begin(set), end(set), [&] () { return udist(rng); });
        std::sort(begin(set), end(set));

        return set;
    }

    ///
    /// \brief randomly sample without replacement the given percentage of `count` elements.
    ///
    /// NB: the returned indices are sorted to potentially improve speed.
    ///
    inline auto sample_without_replacement(const tensor_size_t count, const tensor_size_t percentage)
    {
        assert(0 <= percentage && percentage <= 100);

        indices_t all = indices_t::LinSpaced(count, 0, count);
        std::shuffle(begin(all), end(all), make_rng());

        indices_t set = all.segment(0, percentage * count / 100);
        std::sort(begin(set), end(set));

        return set;
    }
}
