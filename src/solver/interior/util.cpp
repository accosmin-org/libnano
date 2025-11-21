#include <solver/interior/util.h>

using namespace nano;

namespace
{
template <class Qrow, class... Xrows>
auto inorm(const Qrow& qrow, const Xrows&... xrows)
{
    return std::max({qrow.template lpNorm<Eigen::Infinity>(), (xrows.template lpNorm<Eigen::Infinity>(), ...)});
}

template <class Qrow, class Grow, class Arow>
auto delta(const Qrow& qrow, const Grow& grow, const Arow& arow)
{
    const auto Qdelta = (1.0 - qrow.array()).matrix().template lpNorm<Eigen::Infinity>();
    const auto Gdelta = (1.0 - grow.array()).matrix().template lpNorm<Eigen::Infinity>();
    const auto Adelta = (1.0 - arow.array()).matrix().template lpNorm<Eigen::Infinity>();
    return std::max({Qdelta, Gdelta, Adelta});
}

void scale(const scalar_t row_norm, const scalar_t tau, scalar_t& current_scale)
{
    if (row_norm > tau)
    {
        current_scale = 1.0 / std::sqrt(std::min(row_norm, 1.0 / tau));
    }
}
} // namespace

void nano::modified_ruiz_equilibration(vector_t& dQ, matrix_t& Q, vector_t& c, vector_t& dG, matrix_t& G, vector_t& h,
                                       vector_t& dA, matrix_t& A, vector_t& b, const scalar_t tau,
                                       const scalar_t tolerance)
{
    const auto n         = dQ.size();
    const auto m         = dG.size();
    const auto p         = dA.size();
    const auto max_iters = 100;
    const auto is_linear = Q.size() == 0;

    assert(is_linear || Q.rows() == n);
    assert(is_linear || Q.cols() == n);
    assert(c.size() == n);

    assert(G.rows() == m);
    assert(G.cols() == n);
    assert(h.size() == m);

    assert(A.rows() == p);
    assert(A.cols() == n);
    assert(b.size() == p);

    dQ.full(1.0);
    dG.full(1.0);
    dA.full(1.0);

    auto cc = 1.0;
    auto cQ = make_full_vector<scalar_t>(n, 1.0);
    auto cG = make_full_vector<scalar_t>(m, 1.0);
    auto cA = make_full_vector<scalar_t>(p, 1.0);

    for (auto k = 0; k < max_iters && (k == 0 || ::delta(cQ, cG, cA) > tolerance); ++k)
    {
        // matrix equilibration
        for (tensor_size_t i = 0; i < n && !is_linear; ++i)
        {
            if (m > 0 && p > 0)
            {
                ::scale(::inorm(Q.row(i), G.col(i), A.col(i)), tau, cQ(i));
            }
            else if (m > 0)
            {
                ::scale(::inorm(Q.row(i), G.col(i)), tau, cQ(i));
            }
            else if (p > 0)
            {
                ::scale(::inorm(Q.row(i), A.col(i)), tau, cQ(i));
            }
        }
        for (tensor_size_t i = 0; i < m; ++i)
        {
            ::scale(G.row(i).lpNorm<Eigen::Infinity>(), tau, cG(i));
        }
        for (tensor_size_t i = 0; i < p; ++i)
        {
            ::scale(A.row(i).lpNorm<Eigen::Infinity>(), tau, cA(i));
        }

        if (!is_linear)
        {
            Q.matrix().noalias() = cQ.vector().asDiagonal() * Q * cQ.vector().asDiagonal();
        }
        G.matrix().noalias() = cG.vector().asDiagonal() * G * cQ.vector().asDiagonal();
        A.matrix().noalias() = cA.vector().asDiagonal() * A * cQ.vector().asDiagonal();

        c.array() *= cQ.array();
        h.array() *= cG.array();
        b.array() *= cA.array();

        // update equilibration matrices
        if (!is_linear)
        {
            dQ.array() *= cQ.array();
        }
        dG.array() *= cG.array();
        dA.array() *= cA.array();

        // cost scaling
        const auto Qnorm = is_linear ? 0.0 : Q.matrix().rowwise().lpNorm<Eigen::Infinity>().mean();
        const auto cnorm = c.lpNorm<Eigen::Infinity>();
        const auto gamma = 1.0 / std::max(Qnorm, cnorm);

        cc *= gamma;
        Q.array() *= gamma;
        c.array() *= gamma;
    }

    // upscale Lagrange multipliers to recover the original problem
    dG.array() /= cc;
    dA.array() /= cc;
}
