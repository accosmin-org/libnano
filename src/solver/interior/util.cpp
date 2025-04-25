#include <solver/interior/util.h>

#include <iomanip>
#include <iostream>

using namespace nano;

namespace
{
template <class Qrow, class Grow, class Arow>
auto inorm(const Qrow& qrow, const Grow& grow, const Arow& arow)
{
    const auto Qnorm = qrow.template lpNorm<Eigen::Infinity>();
    const auto Gnorm = grow.template lpNorm<Eigen::Infinity>();
    const auto Anorm = arow.template lpNorm<Eigen::Infinity>();
    return std::max({Qnorm, Gnorm, Anorm});
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
        current_scale = 1.0 / std::sqrt(row_norm);
    }
}
} // namespace

scalar_t nano::make_xmax(const vector_t& x, const vector_t& dx, const matrix_t& G, const vector_t& h)
{
    assert(x.size() == dx.size());
    assert(x.size() == G.cols());
    assert(h.size() == G.rows());

    return make_umax(h - G * x, -G * dx);
}

void nano::modified_ruiz_equilibration(vector_t& dQ, matrix_t& Q, vector_t& c, vector_t& dG, matrix_t& G, vector_t& h,
                                       vector_t& dA, matrix_t& A, vector_t& b, const scalar_t tau,
                                       const scalar_t tolerance)
{
    const auto n         = dQ.size();
    const auto m         = dG.size();
    const auto p         = dA.size();
    const auto max_iters = 100;

    assert(Q.rows() == n);
    assert(Q.cols() == n);
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

    auto cQ = make_full_vector<scalar_t>(n, 1.0);
    auto cG = make_full_vector<scalar_t>(m, 1.0);
    auto cA = make_full_vector<scalar_t>(p, 1.0);

    // TODO: the scaling factor `c` from (2) is not working!
    // TODO: the scaling factor `gamma` from (2) is not working!

    for (auto k = 0; k < max_iters && (k == 0 || ::delta(cQ, cG, cA) > tolerance); ++k)
    {
        std::cout << std::setprecision(16) << "k=" << k << ",delta=" << ::delta(cQ, cG, cA) << std::endl;

        for (tensor_size_t i = 0; i < n; ++i)
        {
            ::scale(::inorm(Q.row(i), G.col(i), A.col(i)), tau, cQ(i));
        }
        for (tensor_size_t i = 0; i < m; ++i)
        {
            ::scale(G.row(i).lpNorm<Eigen::Infinity>(), tau, cG(i));
        }
        for (tensor_size_t i = 0; i < p; ++i)
        {
            ::scale(A.row(i).lpNorm<Eigen::Infinity>(), tau, cA(i));
        }

        Q.matrix().noalias() = cQ.vector().asDiagonal() * Q * cQ.vector().asDiagonal();
        G.matrix().noalias() = cG.vector().asDiagonal() * G * cQ.vector().asDiagonal();
        A.matrix().noalias() = cA.vector().asDiagonal() * A * cQ.vector().asDiagonal();

        c.array() *= cQ.array();
        h.array() *= cG.array();
        b.array() *= cA.array();

        /*const auto gamma =
            1.0 / std::max(Q.matrix().rowwise().lpNorm<Eigen::Infinity>().mean(), c.lpNorm<Eigen::Infinity>());

        std::cout << "k=" << k << ",gamma=" << gamma << std::endl;

        Q.array() *= gamma;
        c.array() *= gamma;*/

        dQ.array() *= cQ.array();
        dG.array() *= cG.array();
        dA.array() *= cA.array();
    }
}
