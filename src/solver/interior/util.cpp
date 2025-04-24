#include <solver/interior/util.h>

#include <iostream>

using namespace nano;

namespace
{
template <class Qrow, class Grow, class Arow>
auto delta(const Qrow& qrow, const Grow& grow, const Arow& arow)
{
    return std::max({qrow.template lpNorm<Eigen::Infinity>(), grow.template lpNorm<Eigen::Infinity>(),
                     arow.template lpNorm<Eigen::Infinity>()});
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
                                       vector_t& dA, matrix_t& A, vector_t& b, const scalar_t epsilon)
{
    const auto n = dQ.size();
    const auto m = dG.size();
    const auto p = dA.size();

    assert(Q.rows() == n);
    assert(Q.cols() == n);
    assert(c.size() == n);

    assert(G.rows() == m);
    assert(G.cols() == n);
    assert(h.size() == m);

    assert(A.rows() == p);
    assert(A.cols() == n);
    assert(b.size() == p);

    dQ.full(0.0);
    dG.full(0.0);
    dA.full(0.0);

    auto scale = 1.0;

    for (auto k = 0; k < 100 && 1.0 - delta(dQ, dG, dA) > epsilon; ++k)
    {
        std::cout << "k=" << k << ",delta=" << delta(dQ, dG, dA) << std::endl;

        for (tensor_size_t i = 0; i < n; ++i)
        {
            dQ(i) = 1.0 / std::sqrt(delta(Q.row(i), G.col(i), A.col(i)));
        }

        for (tensor_size_t i = 0; i < m; ++i)
        {
            dG(i) = 1.0 / std::sqrt(G.row(i).lpNorm<Eigen::Infinity>());
        }

        for (tensor_size_t i = 0; i < p; ++i)
        {
            dA(i) = 1.0 / std::sqrt(A.row(i).lpNorm<Eigen::Infinity>());
        }

        std::cout << "k=" << k << ",dQ=" << dQ.transpose() << std::endl;
        std::cout << "k=" << k << ",dG=" << dG.transpose() << std::endl;
        std::cout << "k=" << k << ",dA=" << dA.transpose() << std::endl;
        std::cout << "k=" << k << ",delta=" << delta(dQ, dG, dA) << std::endl;

        Q.matrix().noalias() = dQ.vector().asDiagonal() * Q * dQ.vector().asDiagonal();
        G.matrix().noalias() = dG.vector().asDiagonal() * G * dQ.vector().asDiagonal();
        A.matrix().noalias() = dA.vector().asDiagonal() * A * dQ.vector().asDiagonal();

        c.array() *= dQ.array();
        h.array() *= dG.array();
        b.array() *= dA.array();

        continue;

        Q.array() *= scale;
        c.array() *= scale;

        const auto gamma =
            1.0 / std::max(Q.matrix().rowwise().lpNorm<Eigen::Infinity>().mean(), c.lpNorm<Eigen::Infinity>());

        std::cout << "k=" << k << ",scale=" << scale << ",gamma=" << gamma << std::endl;

        scale *= gamma;
        Q.array() *= gamma;
        c.array() *= gamma;
    }
}
