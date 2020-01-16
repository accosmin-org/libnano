#pragma once

#include <nano/random.h>
#include <nano/function.h>

namespace nano
{
    ///
    /// \brief generic geometric optimization function: f(x) = sum(i, exp(alpha_i + a_i.dot(x))).
    ///
    ///     see "Introductory Lectures on Convex Optimization (Applied Optimization)",
    ///     by Y. Nesterov, 2013, p.56
    ///
    ///     seee "Convex Optimization",
    ///     by S. Boyd and L. Vanderberghe, p.458 (logarithmic version)
    ///
    class function_geometric_optimization_t final : public function_t
    {
    public:

        explicit function_geometric_optimization_t(const tensor_size_t dims, const tensor_size_t summands = 16) :
            function_t("Geometric Optimization", dims, summands, convexity::yes),
            m_a(vector_t::Random(summands)),
            m_A(matrix_t::Random(summands, dims) / dims)
        {
            assert(summands > 0);
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override
        {
            return vgrad(x, m_A, m_a, gx);
        }

        scalar_t vgrad(const vector_t& x, const tensor_size_t begin, const tensor_size_t end, vector_t* gx = nullptr) const override
        {
            assert(0 <= begin && begin < end && end <= summands());

            const auto a = m_a.segment(begin, end - begin);
            const auto A = m_A.block(begin, 0, end - begin, m_A.cols());

            return vgrad(x, A, a, gx);
        }

        void shuffle() const override
        {
            indices_t indices = indices_t::LinSpaced(m_a.size(), 0, m_a.size() - 1);
            std::shuffle(begin(indices), end(indices), make_rng());

            const auto ax = m_a;
            const auto Ax = m_A;
            for (tensor_size_t i = 0; i < indices.size(); ++ i)
            {
                m_a(i) = ax(indices(i));
                m_A.row(i) = Ax.row(indices(i));
            }
        }

    private:

        template <typename tA, typename ta>
        static scalar_t vgrad(const vector_t& x, const tA& A, const ta& a, vector_t* gx = nullptr)
        {
            if (gx != nullptr)
            {
                gx->noalias() = A.transpose() * (a + A * x).array().exp().matrix();
            }

            const scalar_t fx = (a + A * x).array().exp().sum();
            return normalize(fx, gx, a.size());
        }

        static scalar_t normalize(const scalar_t fx, vector_t* gx, const tensor_size_t count)
        {
            if (gx != nullptr && count > 1)
            {
                gx->array() /= static_cast<scalar_t>(count);
            }
            return fx / static_cast<scalar_t>(count);
        }

        // attributes
        mutable vector_t    m_a;
        mutable matrix_t    m_A;
    };
}
