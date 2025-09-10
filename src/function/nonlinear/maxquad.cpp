#include <function/nonlinear/maxquad.h>

using namespace nano;

namespace
{
void fill(tensor2d_map_t A, const tensor_size_t k)
{
    for (tensor_size_t i = 0, dims = A.rows(); i < dims; ++i)
    {
        const auto si = static_cast<scalar_t>(i + 1);
        const auto sk = static_cast<scalar_t>(k + 1);

        for (tensor_size_t j = i + 1; j < dims; ++j)
        {
            const auto sj = static_cast<scalar_t>(j + 1);

            A(i, j) = A(j, i) = std::exp(si / sj) * std::cos(si * sj) * sin(sk);
        }

        auto sum = 0.0;
        for (tensor_size_t j = 0; j < dims; ++j)
        {
            if (i != j)
            {
                sum += std::fabs(A(i, j));
            }
        }

        A(i, i) = si * std::fabs(std::sin(sk)) / static_cast<scalar_t>(dims) + sum;
    }
}

void fill(tensor1d_map_t b, const tensor_size_t k)
{
    for (tensor_size_t i = 0, dims = b.size(); i < dims; ++i)
    {
        const auto si = static_cast<scalar_t>(i + 1);
        const auto sk = static_cast<scalar_t>(k + 1);

        b(i) = std::exp(si / sk) * std::sin(si * sk);
    }
}
} // namespace

function_maxquad_t::function_maxquad_t(const tensor_size_t dims, const tensor_size_t kdims)
    : function_t("maxquad", dims)
    , m_Aks(kdims, dims, dims)
    , m_bks(kdims, dims)
{
    convex(convexity::yes);
    smooth(smoothness::no);
    strong_convexity(0.0);

    for (tensor_size_t k = 0; k < m_Aks.size<0>(); ++k)
    {
        ::fill(m_Aks.tensor(k), k);
        ::fill(m_bks.tensor(k), k);
    }
}

rfunction_t function_maxquad_t::clone() const
{
    return std::make_unique<function_maxquad_t>(*this);
}

scalar_t function_maxquad_t::do_eval(eval_t eval) const
{
    auto kmax = tensor_size_t{0};

    auto fx = std::numeric_limits<scalar_t>::lowest();
    for (tensor_size_t k = 0; k < m_Aks.size<0>(); ++k)
    {
        const auto kfx = x.dot(m_Aks.matrix(k) * x - m_bks.vector(k));
        if (kfx > fx)
        {
            fx   = kfx;
            kmax = k;
        }
    }

    if (gx.size() == x.size())
    {
        gx = 2.0 * m_Aks.matrix(kmax) * x - m_bks.vector(kmax);
    }

    return fx;
}

rfunction_t function_maxquad_t::make(const tensor_size_t dims) const
{
    return std::make_unique<function_maxquad_t>(dims);
}
