#include <solver/interior/minres.h>

#include <iomanip>
#include <iostream>

using namespace nano;

bool nano::MINRES(const matrix_t& A, const vector_t& b, vector_t& x, const tensor_size_t max_iters,
                  const scalar_t tolerance)
{
    auto r  = vector_t{b - A * x};
    auto p0 = vector_t{r};
    auto s0 = vector_t{A * p0};
    auto p1 = vector_t{p0};
    auto s1 = vector_t{s0};
    auto p2 = vector_t{};
    auto s2 = vector_t{};

    for (tensor_size_t iter = 0; iter < max_iters; ++iter)
    {
        p2 = p1;
        p1 = p0;

        s2 = s1;
        s1 = s0;

        const auto alpha = r.dot(s1) / s1.dot(s1);
        if (!std::isfinite(alpha))
        {
            return false;
        }

        x += alpha * p1;
        r -= alpha * s1;

        std::cout << std::setprecision(16) << "iter=" << iter << ": |r|=" << r.lpNorm<Eigen::Infinity>() << "/"
                  << tolerance << ", alpha=" << alpha << std::endl;

        if (r.lpNorm<Eigen::Infinity>() < tolerance)
        {
            return true;
        }

        p0 = s1;
        s0 = A * s1;

        const auto beta1 = s0.dot(s1) / s1.dot(s1);
        p0 -= beta1 * p1;
        s0 -= beta1 * s1;

        if (iter > 0)
        {
            const auto beta2 = s0.dot(s2) / s2.dot(s2);
            p0 -= beta2 * p2;
            s0 -= beta2 * s2;
        }
    }

    return false;
}
