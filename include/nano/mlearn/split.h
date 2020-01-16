#pragma once

#include <set>
#include <nano/random.h>
#include <nano/tensor.h>
#include <nano/mlearn/fold.h>

namespace nano
{
    ///
    /// \brief dataset splitting sample indices into training, validation and test.
    ///
    class split_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        split_t() = default;

        ///
        /// \brief constructor
        ///
        split_t(std::tuple<indices_t, indices_t>&& tr_vd_indices, indices_t te_indices) :
            m_tr_indices(std::move(std::get<0>(tr_vd_indices))),
            m_vd_indices(std::move(std::get<1>(tr_vd_indices))),
            m_te_indices(std::move(te_indices))
        {
        }

        ///
        /// \brief constructor
        ///
        split_t(std::tuple<indices_t, indices_t, indices_t>&& tr_vd_te_indices) :   // NOLINT(hicpp-explicit-conversions)
            m_tr_indices(std::move(std::get<0>(tr_vd_te_indices))),
            m_vd_indices(std::move(std::get<1>(tr_vd_te_indices))),
            m_te_indices(std::move(std::get<2>(tr_vd_te_indices)))
        {
        }

        ///
        /// \returns true if the training, validation and test sample indices
        ///     are valid relative to the given expected number of samples
        ///
        bool valid(const tensor_size_t samples) const
        {
            const auto tr = std::set<tensor_size_t>{begin(m_tr_indices), end(m_tr_indices)};
            const auto vd = std::set<tensor_size_t>{begin(m_vd_indices), end(m_vd_indices)};
            const auto te = std::set<tensor_size_t>{begin(m_te_indices), end(m_te_indices)};

            return  m_tr_indices.size() > 0 &&
                    m_vd_indices.size() > 0 &&
                    m_te_indices.size() > 0 &&
                    (m_tr_indices.minCoeff() >= 0 && m_tr_indices.maxCoeff() < samples) &&
                    (m_vd_indices.minCoeff() >= 0 && m_vd_indices.maxCoeff() < samples) &&
                    (m_te_indices.minCoeff() >= 0 && m_te_indices.maxCoeff() < samples) &&
                    (m_tr_indices.size() + m_vd_indices.size() + m_te_indices.size() == samples) &&
                    (std::find_if(vd.begin(), vd.end(), [&] (const auto i) { return tr.count(i) > 0; }) == vd.end()) &&
                    (std::find_if(te.begin(), te.end(), [&] (const auto i) { return tr.count(i) > 0; }) == te.end());
        }

        ///
        /// \brief returns the sample indices of the given fold.
        ///
        auto& indices(const protocol p)
        {
            switch (p)
            {
            case protocol::train:   return m_tr_indices;
            case protocol::valid:   return m_vd_indices;
            default:                return m_te_indices;
            }
        }
        auto& indices(const fold_t& fold)
        {
            return indices(fold.m_protocol);
        }

        ///
        /// \brief returns the sample indices of the given fold.
        ///
        const auto& indices(const protocol p) const
        {
            switch (p)
            {
            case protocol::train:   return m_tr_indices;
            case protocol::valid:   return m_vd_indices;
            default:                return m_te_indices;
            }
        }
        const auto& indices(const fold_t& fold) const
        {
            return indices(fold.m_protocol);
        }

    private:

        // attributes
        indices_t           m_tr_indices;   ///< indices of the training samples
        indices_t           m_vd_indices;   ///< indices of the validation samples
        indices_t           m_te_indices;   ///< indices of the test samples
    };

    using splits_t = std::vector<split_t>;

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
        indices_t set3 = all.segment(size1 + size2, size3);

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
