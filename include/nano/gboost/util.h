#pragma once

#include <algorithm>
#include <nano/tensor.h>

namespace nano { namespace gboost
{
    ///
    /// \brief min-reduce the given set of per-thread caches using the `min_score` attribute.
    ///
    template <typename tcache>
    const auto& min_reduce(const std::vector<tcache>& caches)
    {
        const auto op = [] (const tcache& one, const tcache& other) { return one.m_score < other.m_score; };
        const auto it = std::min_element(caches.begin(), caches.end(), op);
        return *it;
    }

    ///
    /// \brief map-reduce the given set of per-thread caches into the first cache.
    ///
    template <typename tcache>
    const auto& sum_reduce(std::vector<tcache>& caches, const tensor_size_t samples)
    {
        auto& cache0 = caches[0];
        for (size_t i = 1; i < caches.size(); ++ i)
        {
            cache0 += caches[i];
        }
        return (cache0 /= samples);
    }

    ///
    /// \brief accumulates residuals of different orders useful for fitting simple weak learners.
    ///
    class accumulator_t
    {
    public:

        accumulator_t() = default;

        explicit accumulator_t(const tensor3d_dim_t& tdim) :
            m_r1(tdim),
            m_rx(tdim),
            m_r2(tdim)
        {
        }

        auto& x0() { return m_x0; }
        auto& x1() { return m_x1; }
        auto& x2() { return m_x2; }
        auto r1() { return m_r1.array(); }
        auto rx() { return m_rx.array(); }
        auto r2() { return m_r2.array(); }

        [[nodiscard]] auto x0() const { return m_x0; }
        [[nodiscard]] auto x1() const { return m_x1; }
        [[nodiscard]] auto x2() const { return m_x2; }
        [[nodiscard]] auto r1() const { return m_r1.array(); }
        [[nodiscard]] auto rx() const { return m_rx.array(); }
        [[nodiscard]] auto r2() const { return m_r2.array(); }

        void clear()
        {
            m_r1.zero();
            m_rx.zero();
            m_r2.zero();
            m_x0 = m_x1 = m_x2 = 0.0;
        }

        template <typename tarray>
        void update(scalar_t value, tarray&& vgrad)
        {
            x0() += 1;
            x1() += value;
            x2() += value * value;
            r1() -= vgrad;
            rx() -= vgrad * value;
            r2() += vgrad * vgrad;
        }

        template <typename tarray>
        void update(tarray&& vgrad)
        {
            x0() += 1;
            r1() -= vgrad;
            r2() += vgrad * vgrad;
        }

        // attributes
        scalar_t    m_x0{0}, m_x1{0}, m_x2{0};  ///<
        tensor3d_t  m_r1, m_rx, m_r2;           ///<
    };
}}
