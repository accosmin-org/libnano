#include <nano/core/stats.h>
#include <nano/function/kinks.h>

using namespace nano;

function_kinks_t::function_kinks_t(tensor_size_t dims) :
    benchmark_function_t("Kinks", dims),
    m_kinks(matrix_t::Random(static_cast<tensor_size_t>(std::sqrt(dims)), dims))
{
    convex(true);
    smooth(false);

    vector_t kinks(m_kinks.rows());
    for (tensor_size_t i = 0; i < m_kinks.cols(); ++ i)
    {
        kinks = m_kinks.col(i);

        const auto opt = median(begin(kinks), end(kinks));
            m_offset += (kinks.array() - opt).abs().sum();
    }
}

scalar_t function_kinks_t::vgrad(const vector_t& x, vector_t* gx, vgrad_config_t) const
{
    if (gx != nullptr)
    {
        gx->setZero();
        for (tensor_size_t i = 0; i < m_kinks.rows(); ++ i)
        {
            gx->array() += (x.transpose().array() - m_kinks.row(i).array()).sign();
        }
    }

    scalar_t fx = 0;
    for (tensor_size_t i = 0; i < m_kinks.rows(); ++ i)
    {
        fx += (x.transpose().array() - m_kinks.row(i).array()).abs().sum();
    }
    return fx - m_offset;
}

rfunction_t function_kinks_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_kinks_t>(dims);
}
